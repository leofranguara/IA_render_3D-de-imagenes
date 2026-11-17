import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import slic


def cargar_imagen_cualquier_formato(ruta_imagen: str, max_dim: int = 640) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Carga una imagen en cualquier formato soportado por PIL/OpenCV (PNG, JPG, WEBP, BMP, TIFF, etc.).
    Devuelve (img_bgr, mascara_alfa_existente).
    """
    try:
        with Image.open(ruta_imagen) as pil_img:
            if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
                rgba = np.array(pil_img.convert("RGBA"))
                img_bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
                alfa = rgba[:, :, 3]
            else:
                rgb = np.array(pil_img.convert("RGB"))
                img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                alfa = None
    except Exception:
        data = np.fromfile(ruta_imagen, dtype=np.uint8)
        raw = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen en: {ruta_imagen}")
        if raw.ndim == 2:
            img_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            alfa = None
        elif raw.shape[2] == 4:
            img_bgr = raw[:, :, :3]
            alfa = raw[:, :, 3]
        else:
            img_bgr = raw[:, :, :3]
            alfa = None

    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        escala = max_dim / max(h, w)
        nw, nh = int(w * escala), int(h * escala)
        img_bgr = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        if alfa is not None:
            alfa = cv2.resize(alfa, (nw, nh), interpolation=cv2.INTER_NEAREST)

    mascara_alfa = None
    if alfa is not None and (alfa < 250).mean() > 0.02:
        mascara_alfa = alfa > 127

    return img_bgr, mascara_alfa


def segmentar_objeto_sin_fondo(img_bgr: np.ndarray, mascara_alfa: np.ndarray | None = None) -> np.ndarray:
    """
    Segmentación universal de objeto en primer plano 100% offline.
    Combina:
      1. Conectividad de fondo desde los bordes exteriores (flood-fill / propagación en grafo de
         superpíxeles de baja diferencia de color) para eliminar fondos complejos (como nubes/humo
         arriba y asfalto oscuro abajo) sin tocar la superficie del objeto.
      2. Detección y descarte de sombras proyectadas en el suelo/mesa.
      3. Recuperación de partes oscuras internas del objeto (ruedas/llantas de autos, ventanas,
         parabrisas, parrillas) mediante geometría de arcos inferiores y envolvente local.
      4. Refinamiento de bordes exactos con GrabCut y filtrado guiado.
    """
    if mascara_alfa is not None:
        m = cv2.morphologyEx(mascara_alfa.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        return ndi.binary_fill_holes(m)

    h, w = img_bgr.shape[:2]
    img_suave = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    img_rgb = cv2.cvtColor(img_suave, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_suave, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_suave, cv2.COLOR_BGR2HSV).astype(np.float32)
    gris = cv2.cvtColor(img_suave, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gx = cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gris, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    grad_norm = np.clip(grad / (np.percentile(grad, 98) + 1e-6), 0.0, 1.0)

    # Marco perimetral exterior
    b = max(4, int(min(h, w) * 0.045))
    borde_mask = np.zeros((h, w), dtype=bool)
    borde_mask[:b, :] = True
    borde_mask[-b:, :] = True
    borde_mask[:, :b] = True
    borde_mask[:, -b:] = True

    # Dividir el borde en 8 segmentos perimetrales (esquinas + centros de cada lado)
    # para capturar fondos multimodales (ej. humo blanco en esquina superior izq., fondo negro a la derecha y asfalto abajo)
    mh, mw = h // 2, w // 2
    regiones_borde = [
        lab[:b * 2, :mw, :].reshape(-1, 3),        # superior-izquierda
        lab[:b * 2, mw:, :].reshape(-1, 3),        # superior-derecha
        lab[-b * 2:, :mw, :].reshape(-1, 3),       # inferior-izquierda
        lab[-b * 2:, mw:, :].reshape(-1, 3),       # inferior-derecha
        lab[:mh, :b * 2, :].reshape(-1, 3),        # izquierda-arriba
        lab[mh:, :b * 2, :].reshape(-1, 3),        # izquierda-abajo
        lab[:mh, -b * 2:, :].reshape(-1, 3),       # derecha-arriba
        lab[mh:, -b * 2:, :].reshape(-1, 3),       # derecha-abajo
    ]
    prototipos_bg = []
    for reg in regiones_borde:
        med = np.median(reg, axis=0)
        mad = np.median(np.abs(reg - med), axis=0) + 3.0
        prototipos_bg.append((med, mad))

    # Además, propagar por flood-fill desde las 4 esquinas exteriores las zonas de baja saturación y bajo gradiente
    # (como cielo, paredes lisas o humo/neblina en las esquinas) para garantizar que nunca se tomen como objeto.
    es_fondo_esquina = np.zeros((h, w), dtype=bool)
    semillas = [(2, 2), (2, w - 3), (h - 3, 2), (h - 3, w - 3), (2, mw), (h - 3, mw)]
    img_flood = cv2.GaussianBlur(img_bgr, (7, 7), 0)
    tol = (18, 18, 18)
    for sy, sx in semillas:
        if float(hsv[sy, sx, 1]) < 55.0:  # esquina poco saturada (gris, blanco, negro, asfalto, humo, cielo suave)
            mask_ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(
                img_flood.copy(), mask_ff, (sx, sy), (0, 0, 0),
                loDiff=tol, upDiff=tol, flags=4 | (1 << 8) | cv2.FLOODFILL_MASK_ONLY
            )
            region_ff = mask_ff[1:-1, 1:-1] > 0
            # Si la región inundada no cubre más del 75% de la imagen, es fondo conectado al borde
            if region_ff.mean() < 0.75:
                es_fondo_esquina |= region_ff

    # Distancia mínima de cada píxel a los prototipos de color del borde
    dist_min_bg = np.full((h, w), np.inf, dtype=np.float32)
    dist_ab_min = np.full((h, w), np.inf, dtype=np.float32)
    for med, mad in prototipos_bg:
        d_lab = np.sqrt(
            0.40 * ((lab[:, :, 0] - med[0]) / max(mad[0], 9.0)) ** 2 +
            1.40 * ((lab[:, :, 1] - med[1]) / max(mad[1], 4.5)) ** 2 +
            1.40 * ((lab[:, :, 2] - med[2]) / max(mad[2], 4.5)) ** 2
        )
        d_ab = np.hypot(lab[:, :, 1] - med[1], lab[:, :, 2] - med[2])
        dist_min_bg = np.minimum(dist_min_bg, d_lab)
        dist_ab_min = np.minimum(dist_ab_min, d_ab)

    sat = hsv[:, :, 1]
    sat_borde_p85 = float(np.percentile(sat[borde_mask], 85))

    # Detectar sombras suaves sobre el fondo (menor luminancia que el fondo pero casi el mismo matiz/baja saturación)
    med_global_borde = np.median(lab[borde_mask], axis=0)
    es_sombra = (
        (lab[:, :, 0] < med_global_borde[0] - 8.0) &
        (dist_ab_min < max(11.0, float(np.percentile(dist_ab_min[borde_mask], 90)) * 1.25)) &
        (sat < max(sat_borde_p85 + 22.0, 45.0))
    )

    # Puntuación de pertenencia al objeto principal
    score_fg = (
        (dist_min_bg > 2.3).astype(np.float32) * 0.50 +
        (dist_ab_min > max(12.0, float(np.percentile(dist_ab_min[borde_mask], 92)) * 1.3)).astype(np.float32) * 0.35 +
        (sat > max(sat_borde_p85 + 22.0, 40.0)).astype(np.float32) * 0.25 +
        (grad_norm > 0.28).astype(np.float32) * 0.15
    )
    score_fg[es_sombra] = 0.0
    score_fg[es_fondo_esquina] = 0.0

    fg_inicial = score_fg >= 0.45
    fg_inicial = cv2.morphologyEx(
        fg_inicial.astype(np.uint8), cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ) > 0
    fg_inicial = cv2.morphologyEx(
        fg_inicial.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    ) > 0
    fg_inicial = ndi.binary_fill_holes(fg_inicial) & (~es_fondo_esquina)

    # Refinamiento con GrabCut
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[borde_mask] = cv2.GC_BGD
    gc_mask[es_fondo_esquina] = cv2.GC_BGD

    if fg_inicial.sum() > 0.02 * h * w:
        gc_mask[fg_inicial] = cv2.GC_PR_FGD
        fg_seguro = cv2.erode(
            fg_inicial.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        ) > 0
        if fg_seguro.sum() > 40:
            gc_mask[fg_seguro] = cv2.GC_FGD
        bg_seguro = (dist_min_bg < 1.2) & (~fg_inicial)
        gc_mask[bg_seguro] = cv2.GC_BGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img_bgr, gc_mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
            mascara = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        except Exception:
            mascara = fg_inicial
    else:
        rect = (int(w * 0.06), int(h * 0.06), int(w * 0.88), int(h * 0.88))
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img_bgr, gc_mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
            mascara = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        except Exception:
            mascara = np.ones((h, w), dtype=bool)

    # Limpiar sombras adheridas al contorno exterior sin recortar el núcleo del objeto
    nucleo_objeto = ndi.binary_erosion(mascara, iterations=5)
    mascara = mascara & (~es_sombra | nucleo_objeto)

    # Rescate específico de partes oscuras estructurales (como llantas/ruedas en autos o techos oscuros)
    # Solo rescatamos regiones oscuras que tienen bordes internos fuertes (aros/rines) y están
    # abrazadas por una concavidad semicircular de la carrocería, sin meter manchas blancas de mesa.
    mascara_cerrada = cv2.morphologyEx(
        mascara.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    ) > 0
    en_concavidad = mascara_cerrada & (~mascara) & (~borde_mask)
    es_parte_oscura_con_estructura = (
        en_concavidad &
        (lab[:, :, 0] < 95.0) &
        (grad_norm > 0.16) &
        (~es_sombra)
    )
    if es_parte_oscura_con_estructura.any():
        # Dilatar ligeramente los rines/llantas detectados dentro de la concavidad para cubrir el neumático
        ruedas = cv2.dilate(
            es_parte_oscura_con_estructura.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        ) > 0
        mascara = mascara | (ruedas & mascara_cerrada)

    # Conservar componentes principales y rellenar huecos internos
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mascara.astype(np.uint8), connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = areas.max()
        keep = np.zeros(num_labels, dtype=bool)
        for idx_c, area in enumerate(areas, start=1):
            if area >= max(max_area * 0.14, 0.006 * h * w):
                keep[idx_c] = True
        mascara = keep[labels]

    mascara = cv2.morphologyEx(
        mascara.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ) > 0
    mascara = ndi.binary_fill_holes(mascara)

    if mascara.sum() < 0.02 * h * w:
        return np.ones((h, w), dtype=bool)
    return mascara


def estimar_punto_fuga_hough(img_gris: np.ndarray) -> tuple[float, float, float]:
    """
    Estima el punto de fuga (x_vp, y_vp) normalizado en [0, 1] y una confianza en [0, 1].
    """
    h, w = img_gris.shape
    por_defecto = (0.50, 0.45, 0.0)

    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_gris)
    med = float(np.median(eq))
    bordes = cv2.Canny(eq, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, threshold=40,
                             minLineLength=int(w * 0.08), maxLineGap=10)
    if lineas is None:
        return por_defecto

    L = lineas.reshape(-1, 4).astype(np.float64)
    dx, dy = L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]
    largo = np.hypot(dx, dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        pend = np.abs(dy / dx)
    ok = (np.abs(dx) >= 5.0) & (pend > 0.12) & (pend < 3.0)
    L, largo = L[ok], largo[ok]
    if len(L) < 4:
        return por_defecto

    orden = np.argsort(-largo)[:80]
    L, largo = L[orden], largo[orden]
    n = len(L)

    p1 = np.column_stack([L[:, 0], L[:, 1], np.ones(n)])
    p2 = np.column_stack([L[:, 2], L[:, 3], np.ones(n)])
    rectas = np.cross(p1, p2)
    rectas /= (np.linalg.norm(rectas[:, :2], axis=1, keepdims=True) + 1e-9)

    i, j = np.triu_indices(n, k=1)
    cos_ang = np.abs(np.sum(rectas[i, :2] * rectas[j, :2], axis=1))
    validos = cos_ang < np.cos(np.deg2rad(8))
    i, j = i[validos], j[validos]
    if len(i) == 0:
        return por_defecto

    P = np.cross(rectas[i], rectas[j])
    sano = np.abs(P[:, 2]) > 1e-9
    i, j, P = i[sano], j[sano], P[sano]
    px, py = P[:, 0] / P[:, 2] / w, P[:, 1] / P[:, 2] / h
    peso = largo[i] * largo[j]
    peso_total = peso.sum()

    dentro = (px >= 0.05) & (px <= 0.95) & (py >= 0.05) & (py <= 0.85)
    if dentro.sum() < 3:
        return por_defecto
    px, py, peso = px[dentro], py[dentro], peso[dentro]

    bins = 40
    hist, _, _ = np.histogram2d(py, px, bins=bins, range=[[0, 1], [0, 1]], weights=peso)
    hist = cv2.GaussianBlur(hist.astype(np.float32), (5, 5), 0)
    iy, ix = np.unravel_index(np.argmax(hist), hist.shape)
    cx, cy = (ix + 0.5) / bins, (iy + 0.5) / bins

    cerca = np.hypot(px - cx, py - cy) < 0.10
    if cerca.sum() < 3:
        return por_defecto
    x_vp = float(np.average(px[cerca], weights=peso[cerca]))
    y_vp = float(np.average(py[cerca], weights=peso[cerca]))
    confianza = float(peso[cerca].sum() / (peso_total + 1e-9))

    return (float(np.clip(x_vp, 0.15, 0.85)), float(np.clip(y_vp, 0.15, 0.75)),
            float(np.clip(confianza, 0.0, 1.0)))


def clasificar_tipo_escena(img_bgr: np.ndarray, mascara_alfa: np.ndarray | None,
                           vp_confianza: float, y_horizonte: float) -> tuple[str, bool]:
    """
    Detecta automáticamente si la imagen es:
      - 'objeto': un objeto principal aislable (auto, fruta, persona, mueble, mascota, producto)
      - 'arquitectura': casa, edificio, estructura con líneas geométricas y suelo
      - 'paisaje': paisaje abierto, naturaleza, carretera, montaña, calle completa
    Devuelve (tipo_escena, recomienda_quitar_fondo).
    """
    if mascara_alfa is not None:
        return "objeto", True

    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Evaluar presencia de cielo abierto en el tercio superior
    tercio_sup = lab[: int(h * 0.35), :, :]
    enfoque_sup = cv2.Laplacian(gris[: int(h * 0.35), :], cv2.CV_32F)
    var_sup = float(np.mean(np.abs(enfoque_sup)))
    L_sup = tercio_sup[:, :, 0] / 255.0
    b_sup = tercio_sup[:, :, 2] - 128.0
    frac_cielo = float(((L_sup > 0.52) & (b_sup < 8.0)).mean())

    # Evaluar complejidad del borde vs centro
    b = max(4, int(min(h, w) * 0.06))
    borde_mask = np.zeros((h, w), dtype=bool)
    borde_mask[:b, :] = True
    borde_mask[-b:, :] = True
    borde_mask[:, :b] = True
    borde_mask[:, -b:] = True
    centro_mask = np.zeros((h, w), dtype=bool)
    centro_mask[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)] = True

    sat_centro = float(hsv[:, :, 1][centro_mask].mean())
    sat_borde = float(hsv[:, :, 1][borde_mask].mean())

    # Detectar líneas horizontales y verticales fuertes (típico de casas y edificios)
    bordes = cv2.Canny(gris, 60, 160)
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, threshold=45, minLineLength=int(min(h, w) * 0.12), maxLineGap=8)
    n_ortogonales = 0
    if lineas is not None:
        L = lineas.reshape(-1, 4)
        ang = np.abs(np.rad2deg(np.arctan2(L[:, 3] - L[:, 1], L[:, 2] - L[:, 0])))
        horiz = (ang < 12) | (ang > 168)
        vert = (ang > 78) & (ang < 102)
        n_ortogonales = int(horiz.sum() + vert.sum())

    # Si hay mucho cielo arriba, o perspectiva lineal fuerte con suelo/vegetación en los bordes -> paisaje o casa
    if frac_cielo > 0.28 and var_sup < 18.0 and (sat_centro - sat_borde < 35.0):
        if n_ortogonales >= 12:
            return "arquitectura", False
        return "paisaje", False

    if vp_confianza >= 0.32 and frac_cielo > 0.15:
        return "paisaje", False

    if n_ortogonales >= 18 and (sat_centro - sat_borde < 25.0):
        return "arquitectura", False

    # Probar si hay un objeto bien delimitado que no ocupe toda la foto
    fg_test = segmentar_objeto_sin_fondo(img_bgr)
    frac_fg = float(fg_test.mean())
    toca_bordes = float(fg_test[borde_mask].mean())

    if 0.06 <= frac_fg <= 0.78 and toca_bordes < 0.35:
        return "objeto", True

    if n_ortogonales >= 12:
        return "arquitectura", False
    return "paisaje", False


def procesar_superpixeles(ruta_imagen: str, n_segments: int = 850, compactness: float = 12.0,
                          max_dim: int = 640, modo_fondo: str | bool = "auto"):
    """
    modo_fondo:
      - "auto": detecta automáticamente si conviene aislar el objeto (frutilla, auto, producto)
                o modelar la escena completa (paisaje, casa con entorno, calle).
      - True o "quitar": fuerza la eliminación del fondo para aislar el objeto principal.
      - False o "conservar": conserva toda la escena sin recortar el fondo.
    """
    img_bgr, mascara_alfa = cargar_imagen_cualquier_formato(ruta_imagen, max_dim=max_dim)
    h, w = img_bgr.shape[:2]

    img_filtrada = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=45, sigmaSpace=45)
    img_rgb = cv2.cvtColor(img_filtrada, cv2.COLOR_BGR2RGB)
    img_gris = cv2.cvtColor(img_filtrada, cv2.COLOR_BGR2GRAY)
    img_lab = cv2.cvtColor(img_filtrada, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 1. Punto de fuga / horizonte (con confianza)
    x_vp, y_horizonte, vp_confianza = estimar_punto_fuga_hough(img_gris)

    # 2. Clasificación inteligente de escena y decisión de recorte de fondo
    tipo_escena, recomienda_quitar = clasificar_tipo_escena(img_bgr, mascara_alfa, vp_confianza, y_horizonte)

    if modo_fondo == "auto":
        quitar_fondo_efectivo = recomienda_quitar
    elif isinstance(modo_fondo, str):
        quitar_fondo_efectivo = modo_fondo.lower() in ("quitar", "true", "si", "1")
    else:
        quitar_fondo_efectivo = bool(modo_fondo)

    mascara_fg = None
    if quitar_fondo_efectivo or mascara_alfa is not None:
        mascara_fg = segmentar_objeto_sin_fondo(img_bgr, mascara_alfa)
        vp_confianza = min(vp_confianza, 0.15)
    else:
        # También calculamos una máscara suave del sujeto principal para darle relieve extra dentro de la escena
        mascara_sujeto_escena = segmentar_objeto_sin_fondo(img_bgr, None)
    
    # 3. Gradiente y nitidez local
    L_channel = img_lab[:, :, 0]
    gx = cv2.Sobel(L_channel, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L_channel, cv2.CV_32F, 0, 1, ksize=3)
    gradiente = np.sqrt(gx ** 2 + gy ** 2)
    gradiente_norm = np.clip(gradiente / (np.percentile(gradiente, 99) + 1e-6), 0.0, 1.0)

    lap = np.abs(cv2.Laplacian(L_channel, cv2.CV_32F, ksize=3))
    enfoque = cv2.GaussianBlur(lap, (11, 11), 0)
    enfoque_norm = np.clip(enfoque / (np.percentile(enfoque, 97) + 1e-6), 0.0, 1.0)

    # 4. Superpíxeles SLIC
    etiquetas_slic = slic(img_rgb, n_segments=n_segments, compactness=compactness,
                          sigma=0.5, start_label=0, enforce_connectivity=True)

    # 5. Hipervector por superpíxel
    lab = etiquetas_slic.ravel()
    n_sp = int(lab.max()) + 1
    cnt = np.bincount(lab, minlength=n_sp).astype(np.float64) + 1e-9
    grid_y, grid_x = np.mgrid[0:h, 0:w]

    def media(mapa):
        return np.bincount(lab, weights=mapa.ravel().astype(np.float64), minlength=n_sp) / cnt

    hipervectores = np.column_stack([
        media(grid_x / float(w)), media(grid_y / float(h)),
        media(img_lab[:, :, 0] / 255.0), media(img_lab[:, :, 1] / 255.0), media(img_lab[:, :, 2] / 255.0),
        media(gradiente_norm), media(enfoque_norm),
    ])

    info_escena = {
        "x_vp": x_vp,
        "y_horizonte": y_horizonte,
        "vp_confianza": vp_confianza,
        "tipo_escena": tipo_escena,
        "enfoque_map": enfoque_norm,
        "grad_map": gradiente_norm,
        "lab_map": img_lab,
        "mascara_fg": mascara_fg,
        "mascara_sujeto_escena": mascara_sujeto_escena if mascara_fg is None else mascara_fg,
        "quitar_fondo": bool(mascara_fg is not None),
    }
    return img_rgb, etiquetas_slic, hipervectores, info_escena
