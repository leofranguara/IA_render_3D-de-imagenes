"""
Fase 2: K-Means sobre superpíxeles + Motor Universal de Profundidad 100% Offline.

Adapta automáticamente el modelo de profundidad según la naturaleza de la imagen:
  1. Objetos aislados (autos, frutas, vehículos, figuras, productos):
     Combina perfil volumétrico convexo + orientación/perspectiva del objeto (punto más bajo en el
     suelo está más cerca de la cámara) + facetas estructurales por K-Means (parabrisas, capó,
     puertas, ruedas) + relieve fotométrico libre de picos especulares.
  2. Arquitectura / Casas / Edificios:
     Combina plano de terreno en perspectiva + extrusión vertical de fachadas y muros anclados a su
     base en el suelo + separación de techos y cielo + regularización planar por superpíxeles.
  3. Paisajes / Naturaleza / Calles abiertas:
     Combina perspectiva atmosférica (neblina/desaturación lejana), horizonte adaptativo, gradiente
     de nitidez/textura (defocus), propagación raíz->copa para árboles/montañas y cielo al infinito.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score


# --------------------------------------------------------------------------------------
# Utilidades de filtrado y regularización
# --------------------------------------------------------------------------------------
def filtro_guiado_rapido(guia_rgb: np.ndarray, mapa_z: np.ndarray, radio: int = 6,
                          eps: float = 4e-4) -> np.ndarray:
    """Filtro guiado (He et al.) con la imagen como guía: alinea los bordes de Z con los de la foto."""
    I = cv2.cvtColor(guia_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    p = mapa_z.astype(np.float32)
    k = (2 * radio + 1, 2 * radio + 1)
    mean_I = cv2.boxFilter(I, cv2.CV_32F, k)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, k)
    cov_Ip = cv2.boxFilter(I * p, cv2.CV_32F, k) - mean_I * mean_p
    var_I = cv2.boxFilter(I * I, cv2.CV_32F, k) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return cv2.boxFilter(a, cv2.CV_32F, k) * I + cv2.boxFilter(b, cv2.CV_32F, k)


def _normalizar(mapa: np.ndarray, p_lo: float = 1, p_hi: float = 99) -> np.ndarray:
    lo, hi = np.percentile(mapa, [p_lo, p_hi])
    return np.clip((mapa - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)


def _ajuste_planos_por_superpixel(d: np.ndarray, slic_map: np.ndarray,
                                  ridge: float = 2e-5, max_pend: float = 2.5) -> np.ndarray:
    """
    Ajusta un plano d ≈ c + a·(x-mx) + b·(y-my) dentro de cada superpíxel (Make3D).
    Ideal para facetas de vehículos, paredes de casas y laderas de paisajes.
    """
    H, W = d.shape
    n_sp = int(slic_map.max()) + 1
    lab = slic_map.ravel()
    ys, xs = np.mgrid[0:H, 0:W]
    x = (xs / W).ravel().astype(np.float64)
    y = (ys / H).ravel().astype(np.float64)
    v = d.ravel().astype(np.float64)
    cnt = np.bincount(lab, minlength=n_sp).astype(np.float64) + 1e-9

    def S(arr):
        return np.bincount(lab, weights=arr, minlength=n_sp) / cnt

    mx, my, mv = S(x), S(y), S(v)
    sxx, syy, sxy = S(x * x) - mx ** 2, S(y * y) - my ** 2, S(x * y) - mx * my
    sxv, syv = S(x * v) - mx * mv, S(y * v) - my * mv
    det = (sxx + ridge) * (syy + ridge) - sxy ** 2
    a = np.clip(((syy + ridge) * sxv - sxy * syv) / det, -max_pend, max_pend)
    b = np.clip(((sxx + ridge) * syv - sxy * sxv) / det, -max_pend, max_pend)

    plano = mv[lab] + a[lab] * (x - mx[lab]) + b[lab] * (y - my[lab])
    idx = np.arange(n_sp)
    vmin = ndi.minimum(d, slic_map, idx)
    vmax = ndi.maximum(d, slic_map, idx)
    plano = np.clip(plano, vmin[lab], vmax[lab])
    return plano.reshape(H, W).astype(np.float32)


def _prior_suelo(H: int, W: int, y_horizonte: float) -> np.ndarray:
    y = (np.arange(H, dtype=np.float32) / H).reshape(H, 1)
    g = np.clip((y - y_horizonte) / (1.0 - y_horizonte + 1e-6), 0.0, 1.0)
    return np.repeat(g, W, axis=1)


def _anclar_verticales_al_suelo(d, es_suelo, es_cielo, peso_vertical):
    H, W = d.shape
    base = np.full(W, np.nan, dtype=np.float32)
    d_base = np.full((H, W), np.nan, dtype=np.float32)
    for y in range(H - 1, -1, -1):
        fila_suelo = es_suelo[y]
        base = np.where(fila_suelo, d[y], base)
        d_base[y] = base
    objeto = (~es_suelo) & (~es_cielo) & np.isfinite(d_base) & (d > d_base)
    out = d.copy()
    out[objeto] = d_base[objeto] + (1.0 - peso_vertical) * (d[objeto] - d_base[objeto])
    return out


def _ajuste_robusto(x: np.ndarray, y: np.ndarray, iters: int = 4):
    mask = np.ones_like(x, dtype=bool)
    a, b = 1.0, 0.0
    for _ in range(iters):
        if mask.sum() < 50:
            break
        a, b = np.polyfit(x[mask], y[mask], 1)
        res = np.abs(y - (a * x + b))
        umbral = max(1.5 * 1.4826 * np.median(res[mask]), 0.02)
        mask = res < umbral
    return float(a), float(b), float(mask.mean())


def _mascara_cielo(img_lab, enfoque, y_norm, y_horizonte, disp=None):
    L = img_lab[:, :, 0] / 255.0
    b_c = img_lab[:, :, 2] - 128.0
    a_c = img_lab[:, :, 1] - 128.0
    # Cielo claro (azul, celeste o nublado uniforme en la parte superior)
    cielo = (L > 0.54) & (b_c < 10.0) & (a_c > -12.0) & (enfoque < 0.30) & (y_norm < y_horizonte + 0.06)
    if disp is not None:
        cielo &= disp < np.percentile(disp, 35)
    cielo = cv2.morphologyEx(cielo.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, comp, stats, _ = cv2.connectedComponentsWithStats(cielo, connectivity=8)
    grande = np.zeros(n, dtype=bool)
    grande[1:] = stats[1:, cv2.CC_STAT_AREA] > 0.01 * cielo.size
    return grande[comp]


# --------------------------------------------------------------------------------------
# 1. Profundidad para Objetos Aislados (Autos, Frutas, Casas recortadas, Productos)
# --------------------------------------------------------------------------------------
def _profundidad_volumetrica_objeto(
    img_rgb: np.ndarray,
    img_lab: np.ndarray,
    enfoque: np.ndarray,
    grad_map: np.ndarray,
    mascara_fg: np.ndarray,
    mapa_clusters: np.ndarray,
) -> np.ndarray:
    """
    Reconstruye el volumen 3D de cualquier objeto aislado (orgánico como una fruta o
    estructurado/geométrico como un automóvil, casa o mueble):
      - Transformada de distancia multi-escala (cúpula elipsoidal suave + cresta estructural).
      - Inclinación de perspectiva vertical/horizontal a partir del contorno inferior del objeto
        (en autos o casas vistas en 3/4, la esquina más baja en la foto es el parachoques/esquina
        más cercana a la cámara).
      - Coherencia por facetas K-Means (capó, parabrisas, ventanas laterales, puertas).
      - Sombreado suave (Shape-from-Shading) con supresión de brillos especulares.
    """
    H, W = mascara_fg.shape
    m_u8 = mascara_fg.astype(np.uint8)
    dist = cv2.distanceTransform(m_u8, cv2.DIST_L2, 5).astype(np.float32)
    max_d = float(dist.max())
    if max_d < 1e-3:
        return np.zeros_like(enfoque, dtype=np.float32)

    # A. Base volumétrica suave (perfil elipsoidal que llega limpio al contorno)
    r = np.clip(dist / max_d, 0.0, 1.0)
    domo_eliptico = np.sqrt(np.clip(2.0 * r - r * r, 0.0, 1.0))
    domo_suave = cv2.GaussianBlur(domo_eliptico, (0, 0), sigmaX=max(3.0, max_d * 0.12))
    if domo_suave[mascara_fg].max() > 1e-6:
        domo_suave /= float(domo_suave[mascara_fg].max())

    # B. Rampa de perspectiva 3D desde la huella inferior del objeto
    # En cualquier objeto apoyado (auto, casa, mueble), la parte más baja de cada columna
    # indica qué tan cerca está esa sección respecto al observador.
    ys_fg, xs_fg = np.where(mascara_fg)
    y_min, y_max = int(ys_fg.min()), int(ys_fg.max())
    x_min, x_max = int(xs_fg.min()), int(xs_fg.max())
    alto_obj = max(y_max - y_min, 1)
    ancho_obj = max(x_max - x_min, 1)

    # Encontrar el borde inferior del objeto por columna y suavizarlo
    hay_col = mascara_fg.any(axis=0)
    fila_inf_col = H - 1 - np.argmax(mascara_fg[::-1, :], axis=0)
    cols_validas = np.where(hay_col)[0]
    if len(cols_validas) > 5:
        y_base_col = np.interp(
            np.arange(W),
            cols_validas,
            fila_inf_col[cols_validas].astype(np.float32),
        )
        y_base_col = cv2.GaussianBlur(y_base_col.reshape(1, W), (1, 31), 0).ravel()
        # Normalizar entre 0 y 1 según qué parte de la base baja más en la foto (más cerca)
        b_min, b_max = float(y_base_col[cols_validas].min()), float(y_base_col[cols_validas].max())
        if b_max - b_min > 0.08 * alto_obj:
            persp_x = np.clip((y_base_col - b_min) / (b_max - b_min + 1e-6), 0.0, 1.0)
        else:
            persp_x = np.full(W, 0.5, dtype=np.float32)
    else:
        persp_x = np.full(W, 0.5, dtype=np.float32)

    grid_y, _ = np.mgrid[0:H, 0:W]
    rel_y = np.clip((grid_y - y_min) / float(alto_obj), 0.0, 1.0)
    # La parte inferior-frontal del objeto sobresale ligeramente más que el techo trasero
    rampa_perspectiva = 0.55 * persp_x[None, :] + 0.45 * (1.0 - 0.35 * (1.0 - rel_y))
    rampa_perspectiva = cv2.GaussianBlur(rampa_perspectiva.astype(np.float32), (15, 15), 0)

    # C. Relieve por facetas K-Means y luminancia sin distorsión por reflejos o vidrios oscuros
    L = img_lab[:, :, 0] / 255.0
    # Atenuar reflejos especulares blancos y no hundir superficies oscuras (como vidrios polarizados o llantas)
    L_templada = np.clip(L, 0.25, 0.82)
    L_templada = (L_templada - 0.25) / 0.57
    L_suave = cv2.GaussianBlur(L_templada.astype(np.float32), (11, 11), 0)

    # Promedio por cluster dentro del objeto para dar estructura de paneles (capó, puertas, hojas)
    n_k = int(mapa_clusters.max()) + 1
    z_base_combinada = 0.68 * domo_suave + 0.22 * rampa_perspectiva + 0.10 * L_suave
    cnt_k = np.bincount(mapa_clusters[mascara_fg].ravel(), minlength=n_k).astype(np.float64) + 1e-9
    sum_k = np.bincount(
        mapa_clusters[mascara_fg].ravel(),
        weights=z_base_combinada[mascara_fg].ravel().astype(np.float64),
        minlength=n_k,
    )
    media_k = (sum_k / cnt_k).astype(np.float32)
    mapa_facetas = cv2.GaussianBlur(media_k[mapa_clusters], (9, 9), 0)

    # Combinar volumen global (75%) + estructura de facetas (15%) + micro-detalle (10%)
    detalle = 0.6 * L_suave + 0.4 * cv2.GaussianBlur(enfoque, (7, 7), 0)
    z_interno = (0.72 * z_base_combinada + 0.18 * mapa_facetas + 0.10 * detalle) * np.sqrt(
        np.clip(domo_eliptico * 1.15, 0.0, 1.0)
    )

    z = np.zeros((H, W), dtype=np.float32)
    z[mascara_fg] = np.clip(0.20 + 0.80 * z_interno[mascara_fg], 0.18, 1.0)
    return z


# --------------------------------------------------------------------------------------
# 2. Profundidad para Escenas Completas: Paisajes, Casas, Calles y Vehículos en Escena
# --------------------------------------------------------------------------------------
def _profundidad_escena_universal(
    img_rgb: np.ndarray,
    img_lab: np.ndarray,
    enfoque: np.ndarray,
    grad_map: np.ndarray,
    y_norm: np.ndarray,
    x_norm: np.ndarray,
    y_horizonte: float,
    x_vp: float,
    es_cielo: np.ndarray,
    vp_confianza: float,
    tipo_escena: str,
    mascara_sujeto: np.ndarray | None = None,
) -> np.ndarray:
    """
    Motor de profundidad monocular multi-pista para paisajes, arquitectura/casas y escenas exteriores:
      1. Pista de perspectiva atmosférica / Dark Channel (objetos lejanos pierden contraste y se vuelven
         más azulados/claros por dispersión atmosférica).
      2. Pista de geometría de suelo + cono hacia el punto de fuga (carreteras, valles, calles).
      3. Pista de nitidez y gradiente local (texturas cercanas tienen mayor frecuencia espacial).
      4. Anclaje vertical raíz->techo/copa para casas, edificios, árboles y postes.
      5. Realce volumétrico del sujeto principal (si hay un auto o casa protagonista en la escena).
    """
    H, W = y_norm.shape
    L = img_lab[:, :, 0] / 255.0
    a_c = img_lab[:, :, 1] - 128.0
    b_c = img_lab[:, :, 2] - 128.0

    # --- Pista 1: Perspectiva atmosférica (Haze / Dark Channel Prior) ---
    # En paisajes, las montañas y fondos lejanos tienen alto brillo mínimo en RGB y menor saturación
    img_f = img_rgb.astype(np.float32) / 255.0
    canal_oscuro = np.min(img_f, axis=2)
    canal_oscuro = cv2.erode(canal_oscuro, np.ones((7, 7), np.uint8))
    cercania_atmosferica = 1.0 - cv2.GaussianBlur(canal_oscuro, (15, 15), 0)
    # Tonos cálidos (rojos/amarillos/verdes saturados) suelen estar más cerca que tonos azulados/grisáceos
    calidez = np.clip((b_c - a_c * 0.3 + 20.0) / 60.0, 0.0, 1.0)
    pista_atmosfera = np.clip(0.70 * cercania_atmosferica + 0.30 * calidez, 0.0, 1.0)

    # --- Pista 2: Geometría del plano de tierra y perspectiva hacia el punto de fuga ---
    # Si hay cielo detectado, ajustamos el horizonte real al límite inferior del cielo
    if es_cielo.any():
        filas_cielo = np.where(es_cielo.mean(axis=1) > 0.25)[0]
        if len(filas_cielo) > 0:
            y_h_efectivo = float(np.clip(filas_cielo.max() / float(H), 0.15, 0.65))
        else:
            y_h_efectivo = y_horizonte
    else:
        y_h_efectivo = y_horizonte

    z_suelo = _prior_suelo(H, W, y_h_efectivo - 0.03)
    # Añadir leve curvatura lateral hacia el punto de fuga cuando hay perspectiva vial/arquitectónica
    efecto_vp = np.abs(x_norm - x_vp) * (1.0 - y_norm) * (0.25 if vp_confianza > 0.25 else 0.10)
    z_perspectiva = np.clip(z_suelo * 0.90 + efecto_vp + 0.06, 0.04, 1.0)

    # Por encima del horizonte pero que NO es cielo (montañas, cordilleras, edificios lejanos, árboles):
    # la profundidad decrece suavemente desde la base en el horizonte hacia las cumbres
    sobre_horiz = (y_norm <= y_h_efectivo) & (~es_cielo)
    dist_sobre_h = np.clip((y_h_efectivo - y_norm) / max(y_h_efectivo, 0.15), 0.0, 1.0)
    z_montanas = np.clip(0.28 * (1.0 - 0.65 * dist_sobre_h) + 0.25 * pista_atmosfera, 0.06, 0.55)
    z_base_escena = np.where(sobre_horiz, z_montanas, z_perspectiva)

    # --- Pista 3: Detección y anclaje de estructuras verticales (Casas, Autos, Árboles) ---
    gx = cv2.Sobel(L.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    borde_v = np.abs(gx) / (np.percentile(np.abs(gx), 98) + 1e-6)
    contraste_local = cv2.GaussianBlur(grad_map, (9, 9), 0)

    es_estructura = (~es_cielo) & (
        ((y_norm < y_h_efectivo + 0.03) & (pista_atmosfera > 0.42)) |
        ((y_norm >= y_h_efectivo - 0.02) & (y_norm < 0.92) & (
            (enfoque > 0.28) & (borde_v > 0.24) |
            (contraste_local > 0.26) |
            ((a_c < -8.0) & (enfoque > 0.22))
        ))
    )
    es_estructura = cv2.morphologyEx(
        es_estructura.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 17))
    ) > 0
    es_estructura &= (~es_cielo)

    # Propagar la profundidad de la base donde la estructura toca el suelo hacia arriba (paredes planas en Z)
    base_valida = es_estructura & (y_norm >= y_h_efectivo - 0.02)
    hay_base = base_valida.any(axis=0)
    fila_base = H - 1 - np.argmax(base_valida[::-1, :], axis=0)
    z_raiz_col = np.where(hay_base, z_perspectiva[fila_base, np.arange(W)], z_base_escena.mean(axis=0)).astype(np.float32)
    # Suavizar horizontalmente la base para que las fachadas de casas no tengan estrías verticales
    z_raiz_col = cv2.GaussianBlur(z_raiz_col.reshape(1, W), (1, 21), 0).ravel()
    z_estruct = np.where(es_estructura, z_raiz_col[None, :], 0.0).astype(np.float32)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    z_prop = z_estruct.copy()
    for _ in range(12):
        z_dil = cv2.dilate(z_prop, kern)
        z_prop = np.where(es_estructura, np.maximum(z_prop, z_dil * 0.994), 0.0)

    # Combinar terreno/montañas con estructuras verticales y pista atmosférica
    z = 0.65 * z_base_escena + 0.25 * pista_atmosfera + 0.10 * cv2.GaussianBlur(enfoque, (11, 11), 0)
    if es_estructura.any():
        peso_est = 0.75 if tipo_escena == "arquitectura" else 0.55
        z[es_estructura] = (
            (1.0 - peso_est) * z[es_estructura] +
            peso_est * np.maximum(z_prop[es_estructura], z[es_estructura])
        )

    # Si hay un sujeto destacado en la escena (ej. un auto o una casa principal conservando el fondo),
    # darle cohesión volumétrica para que no se aplane contra el fondo
    if mascara_sujeto is not None:
        frac_s = float(mascara_sujeto.mean())
        if 0.04 <= frac_s <= 0.75:
            dist_s = cv2.distanceTransform(mascara_sujeto.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
            domo_s = np.sqrt(np.clip(dist_s / (float(dist_s.max()) + 1e-6), 0.0, 1.0))
            domo_s = cv2.GaussianBlur(domo_s, (11, 11), 0)
            # Anclar el sujeto a la profundidad del suelo en su punto de apoyo inferior
            ys_s = np.where(mascara_sujeto)[0]
            y_apoyo = float(np.percentile(ys_s, 95)) / float(H)
            z_apoyo = float(np.clip((y_apoyo - y_h_efectivo) / (1.0 - y_h_efectivo + 1e-6), 0.35, 0.85))
            z_sujeto = np.clip(z_apoyo + 0.22 * domo_s, 0.25, 0.98)
            z = np.where(mascara_sujeto, 0.45 * z + 0.55 * np.maximum(z, z_sujeto), z)

    z[es_cielo] = 0.0
    return cv2.GaussianBlur(z.astype(np.float32), (5, 5), 0)


# --------------------------------------------------------------------------------------
# Función principal
# --------------------------------------------------------------------------------------
def agrupar_e_inferir_profundidad(
    img_rgb: np.ndarray,
    slic_map: np.ndarray,
    hipervectores: np.ndarray,
    info_escena: dict,
    n_clusters: int = 8,
    w_espacial: float = 0.30,
    w_color: float = 2.3,
    w_grad: float = 1.2,
    disp_neural: np.ndarray | None = None,
    peso_suelo: float = 0.35,
    peso_vertical: float = 0.85,
    conf_min_perspectiva: float = 0.35,
):
    H, W = slic_map.shape
    y_h = info_escena["y_horizonte"]
    x_vp = info_escena["x_vp"]
    vp_conf = info_escena.get("vp_confianza", 1.0)
    tipo_escena = info_escena.get("tipo_escena", "escena")
    img_lab = info_escena["lab_map"]
    enfoque = info_escena["enfoque_map"]
    grad_map = info_escena["grad_map"]
    mascara_fg = info_escena.get("mascara_fg", None)
    mascara_sujeto_escena = info_escena.get("mascara_sujeto_escena", None)

    # 1. K-Means sobre el hipervector de superpíxeles
    X = hipervectores[:, :6].copy()
    X[:, 0:2] *= w_espacial
    X[:, 2:5] *= w_color
    X[:, 5] *= w_grad
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_sp = kmeans.fit_predict(X)
    silueta = silhouette_score(X, labels_sp)
    dbi = davies_bouldin_score(X, labels_sp)
    mapa_clusters = labels_sp[slic_map]

    grid_y, grid_x = np.mgrid[0:H, 0:W]
    y_norm = (grid_y / float(H)).astype(np.float32)
    x_norm = (grid_x / float(W)).astype(np.float32)

    usa_suelo = False
    if disp_neural is not None:
        modo = "neural"
        d = filtro_guiado_rapido(img_rgb, disp_neural, radio=4, eps=1e-3)
        d = np.clip(d, 0.0, 1.0)
        es_cielo = _mascara_cielo(img_lab, enfoque, y_norm, y_h, d)
        d[es_cielo] = 0.0

        if mascara_fg is None and vp_conf >= conf_min_perspectiva:
            g = _prior_suelo(H, W, y_h)
            zona = (y_norm > max(y_h + 0.10, 0.6)) & (~es_cielo)
            a, b, _ = _ajuste_robusto(g[zona][::7], d[zona][::7])
            if a > 0.05:
                g_al = np.clip(a * g + b, 0.0, 1.0)
                es_suelo = (y_norm > y_h) & (np.abs(d - g_al) < 0.08) & (~es_cielo)
                es_suelo = cv2.morphologyEx(
                    es_suelo.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
                ) > 0
                d = np.where(es_suelo, d + peso_suelo * (g_al - d), d)
                d = _anclar_verticales_al_suelo(d, es_suelo, es_cielo, peso_vertical)
                usa_suelo = True
        if mascara_fg is not None:
            z_domo = _profundidad_volumetrica_objeto(
                img_rgb, img_lab, enfoque, grad_map, mascara_fg, mapa_clusters
            )
            d = np.where(mascara_fg, 0.55 * d + 0.45 * z_domo, 0.0)
        mapa_geom = d
    else:
        modo = f"universal ({tipo_escena})"
        if mascara_fg is not None:
            es_cielo = ~mascara_fg
            mapa_geom = _profundidad_volumetrica_objeto(
                img_rgb, img_lab, enfoque, grad_map, mascara_fg, mapa_clusters
            )
        else:
            es_cielo = _mascara_cielo(img_lab, enfoque, y_norm, y_h)
            usa_suelo = True
            mapa_geom = _profundidad_escena_universal(
                img_rgb, img_lab, enfoque, grad_map, y_norm, x_norm,
                y_h, x_vp, es_cielo, vp_conf, tipo_escena,
                mascara_sujeto=mascara_sujeto_escena,
            )

    # 3. Regularización por superpíxeles: planos inclinados (Make3D) + mezcla suave
    planos = _ajuste_planos_por_superpixel(mapa_geom, slic_map)
    peso_planos = 0.45 if tipo_escena == "arquitectura" else 0.35
    mapa_z_bruto = (1.0 - peso_planos) * mapa_geom + peso_planos * planos

    # 4. Filtro guiado para alinear bordes de profundidad con la silueta real de cada elemento
    mapa_z_final = filtro_guiado_rapido(img_rgb, mapa_z_bruto, radio=6, eps=5e-4)
    if mascara_fg is not None:
        if mascara_fg.any():
            lo, hi = np.percentile(mapa_z_final[mascara_fg], [2, 99])
            z_fg = np.clip((mapa_z_final - lo) / (hi - lo + 1e-6), 0.0, 1.0)
            dist_borde = cv2.distanceTransform(mascara_fg.astype(np.uint8), cv2.DIST_L2, 5)
            peso_borde = np.clip(dist_borde / max(float(dist_borde.max()) * 0.20, 3.0), 0.0, 1.0)
            mapa_z_final = np.where(mascara_fg, 0.20 + 0.80 * z_fg * np.sqrt(peso_borde), 0.0).astype(np.float32)
        else:
            mapa_z_final = _normalizar(mapa_z_final)
    else:
        mapa_z_final = _normalizar(mapa_z_final)
        mapa_z_final[es_cielo] = np.minimum(mapa_z_final[es_cielo], 0.03)

    metricas = {
        "modo_profundidad": modo,
        "tipo_escena": tipo_escena,
        "suelo_en_perspectiva": usa_suelo,
        "fondo_eliminado": mascara_fg is not None,
        "silueta": round(float(silueta), 4),
        "davies_bouldin": round(float(dbi), 4),
        "wcss": round(float(kmeans.inertia_), 2),
        "k_clusters": n_clusters,
        "n_superpixeles": len(hipervectores),
        "y_horizonte": round(float(y_h), 3),
        "x_vp": round(float(x_vp), 3),
        "vp_confianza": round(float(vp_conf), 3),
    }
    return mapa_clusters, mapa_z_bruto, mapa_z_final, metricas
