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
    Extrae la máscara binaria (True = primer plano / objeto, False = fondo) de la imagen.
    Si ya trae canal alfa con transparencia real, la usa directamente.
    Si no, combina análisis del color de borde, saturación, contraste y GrabCut automático.
    """
    if mascara_alfa is not None:
        m = cv2.morphologyEx(mascara_alfa.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        return ndi.binary_fill_holes(m)

    h, w = img_bgr.shape[:2]
    img_suave = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    lab = cv2.cvtColor(img_suave, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_suave, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Muestra el borde perimetral de la imagen (donde casi siempre está el fondo)
    b = max(4, int(min(h, w) * 0.05))
    borde_mask = np.zeros((h, w), dtype=bool)
    borde_mask[:b, :] = True
    borde_mask[-b:, :] = True
    borde_mask[:, :b] = True
    borde_mask[:, -b:] = True

    borde_lab = lab[borde_mask]
    med_lab = np.median(borde_lab, axis=0)
    mad_lab = np.median(np.abs(borde_lab - med_lab), axis=0) + 2.5

    # Distancia robusta de color y crominanacia respecto al fondo del borde
    dist_lab = np.sqrt(
        0.45 * ((lab[:, :, 0] - med_lab[0]) / max(mad_lab[0], 8.0)) ** 2 +
        1.35 * ((lab[:, :, 1] - med_lab[1]) / max(mad_lab[1], 4.0)) ** 2 +
        1.35 * ((lab[:, :, 2] - med_lab[2]) / max(mad_lab[2], 4.0)) ** 2
    )
    dist_ab = np.hypot(lab[:, :, 1] - med_lab[1], lab[:, :, 2] - med_lab[2])

    sat = hsv[:, :, 1]
    sat_borde_p85 = float(np.percentile(sat[borde_mask], 85))

    gris = cv2.cvtColor(img_suave, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gris, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    grad_norm = grad / (np.percentile(grad, 98) + 1e-6)

    # Puntuación inicial de pertenencia al objeto
    score_fg = (
        (dist_lab > 2.4).astype(np.float32) * 0.45 +
        (dist_ab > max(12.0, np.percentile(dist_ab[borde_mask], 92) * 1.3)).astype(np.float32) * 0.35 +
        (sat > max(sat_borde_p85 + 22.0, 40.0)).astype(np.float32) * 0.25 +
        (grad_norm > 0.25).astype(np.float32) * 0.15
    )

    # Descartar sombras suaves sobre fondos claros (baja luminancia pero baja diferencia cromática)
    es_sombra = (
        (lab[:, :, 0] < med_lab[0] - 10.0) &
        (dist_ab < max(11.0, np.percentile(dist_ab[borde_mask], 90) * 1.25)) &
        (sat < max(sat_borde_p85 + 25.0, 48.0))
    )
    score_fg[es_sombra] = 0.0

    fg_inicial = score_fg >= 0.45
    fg_inicial = cv2.morphologyEx(fg_inicial.astype(np.uint8), cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    fg_inicial = cv2.morphologyEx(fg_inicial.astype(np.uint8), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
    fg_inicial = ndi.binary_fill_holes(fg_inicial)

    # Preparar máscara trimap para GrabCut
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[borde_mask] = cv2.GC_BGD

    if fg_inicial.sum() > 0.02 * h * w:
        gc_mask[fg_inicial] = cv2.GC_PR_FGD
        fg_seguro = cv2.erode(fg_inicial.astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        if fg_seguro.sum() > 50:
            gc_mask[fg_seguro] = cv2.GC_FGD
        bg_seguro = (dist_lab < 1.1) & (~fg_inicial)
        gc_mask[bg_seguro] = cv2.GC_BGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img_bgr, gc_mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
            mascara = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        except Exception:
            mascara = fg_inicial
    else:
        # Fallback por rectángulo central si la imagen tiene poco contraste de color con el borde
        rect = (int(w * 0.06), int(h * 0.06), int(w * 0.88), int(h * 0.88))
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img_bgr, gc_mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
            mascara = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        except Exception:
            mascara = np.ones((h, w), dtype=bool)

    # Quitar sombras residuales pegadas al borde del objeto
    mascara = mascara & (~es_sombra | ndi.binary_erosion(mascara, iterations=5))

    # Conservar componentes principales significativas y rellenar huecos internos
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mascara.astype(np.uint8), connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = areas.max()
        keep = np.zeros(num_labels, dtype=bool)
        for idx_c, area in enumerate(areas, start=1):
            if area >= max(max_area * 0.12, 0.005 * h * w):
                keep[idx_c] = True
        mascara = keep[labels]

    mascara = cv2.morphologyEx(mascara.astype(np.uint8), cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    mascara = ndi.binary_fill_holes(mascara)

    # Si por alguna razón quedó vacía, devolver toda la imagen
    if mascara.sum() < 0.01 * h * w:
        return np.ones((h, w), dtype=bool)
    return mascara



def estimar_punto_fuga_hough(img_gris: np.ndarray) -> tuple[float, float, float]:
    """
    Estima el punto de fuga (x_vp, y_vp) normalizado en [0, 1] y una confianza en [0, 1].

    Mejoras respecto a la versión anterior:
      - CLAHE + umbrales de Canny automáticos (según la mediana), robustos a la iluminación.
      - Todas las intersecciones entre las líneas más largas votan con peso = largo_i * largo_j,
        en vez de usar la mediana (que se contaminaba con líneas que no convergen).
      - Se busca el pico de un histograma 2D suavizado y se refina con la media ponderada local.
      - Devuelve una confianza: si la escena NO tiene perspectiva lineal clara (retratos,
        objetos sobre una mesa...) la confianza es baja y las fases siguientes no fuerzan
        un modelo de suelo.
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
    ok = (np.abs(dx) >= 5.0) & (pend > 0.12) & (pend < 3.0)   # descarta horizontales/verticales
    L, largo = L[ok], largo[ok]
    if len(L) < 4:
        return por_defecto

    orden = np.argsort(-largo)[:80]
    L, largo = L[orden], largo[orden]
    n = len(L)

    p1 = np.column_stack([L[:, 0], L[:, 1], np.ones(n)])
    p2 = np.column_stack([L[:, 2], L[:, 3], np.ones(n)])
    rectas = np.cross(p1, p2)                      # rectas en coordenadas homogéneas
    rectas /= (np.linalg.norm(rectas[:, :2], axis=1, keepdims=True) + 1e-9)

    i, j = np.triu_indices(n, k=1)
    # ángulo mínimo entre rectas (evita intersecciones casi paralelas, inestables)
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


def procesar_superpixeles(ruta_imagen: str, n_segments: int = 850, compactness: float = 12.0,
                          max_dim: int = 640, quitar_fondo: bool = False):
    img_bgr, mascara_alfa = cargar_imagen_cualquier_formato(ruta_imagen, max_dim=max_dim)
    h, w = img_bgr.shape[:2]

    mascara_fg = None
    if quitar_fondo or mascara_alfa is not None:
        mascara_fg = segmentar_objeto_sin_fondo(img_bgr, mascara_alfa)

    img_filtrada = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=45, sigmaSpace=45)
    img_rgb = cv2.cvtColor(img_filtrada, cv2.COLOR_BGR2RGB)
    img_gris = cv2.cvtColor(img_filtrada, cv2.COLOR_BGR2GRAY)
    img_lab = cv2.cvtColor(img_filtrada, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 1. Punto de fuga / horizonte (con confianza)
    x_vp, y_horizonte, vp_confianza = estimar_punto_fuga_hough(img_gris)
    if mascara_fg is not None:
        # Si se aisla un objeto sin fondo, no forzamos perspectiva de suelo lineal
        vp_confianza = min(vp_confianza, 0.15)

    # 2. Gradiente y nitidez local
    L_channel = img_lab[:, :, 0]
    gx = cv2.Sobel(L_channel, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L_channel, cv2.CV_32F, 0, 1, ksize=3)
    gradiente = np.sqrt(gx ** 2 + gy ** 2)
    gradiente_norm = np.clip(gradiente / (np.percentile(gradiente, 99) + 1e-6), 0.0, 1.0)

    lap = np.abs(cv2.Laplacian(L_channel, cv2.CV_32F, ksize=3))
    enfoque = cv2.GaussianBlur(lap, (11, 11), 0)
    enfoque_norm = np.clip(enfoque / (np.percentile(enfoque, 97) + 1e-6), 0.0, 1.0)

    # 3. Superpíxeles SLIC
    etiquetas_slic = slic(img_rgb, n_segments=n_segments, compactness=compactness,
                          sigma=0.5, start_label=0, enforce_connectivity=True)

    # 4. Hipervector por superpíxel (vectorizado con bincount: antes se llamaba a ndimage 7 veces)
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
        "enfoque_map": enfoque_norm,
        "grad_map": gradiente_norm,
        "lab_map": img_lab,
        "mascara_fg": mascara_fg,
        "quitar_fondo": bool(quitar_fondo or mascara_alfa is not None),
    }
    return img_rgb, etiquetas_slic, hipervectores, info_escena
