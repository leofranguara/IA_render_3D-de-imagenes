"""
Fase 2: K-Means sobre superpíxeles + mapa de profundidad.

Convención del mapa Z (igual que tu versión original): valores en [0, 1], 1 = CERCA, 0 = LEJOS.
En realidad es una disparidad normalizada; la fase 3 la convierte a profundidad métrica con
un modelo de cámara pinhole.

Dos modos:
  - "neural":     MiDaS aporta la profundidad relativa; aquí se corrige y regulariza con
                  (a) modelo de suelo en perspectiva (si hay punto de fuga fiable),
                  (b) cielo al infinito, (c) planos por superpíxel, (d) filtro guiado.
  - "geometrico": tu heurística original (horizonte + color + textura), corregida y vectorizada.
                  Se usa si no hay modelo neuronal disponible.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


# --------------------------------------------------------------------------------------
# Utilidades
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
                                  ridge: float = 2e-5, max_pend: float = 3.0) -> np.ndarray:
    """
    Ajusta un plano  d ≈ c + a·(x-mx) + b·(y-my)  dentro de cada superpíxel (idea de Make3D).
    Resultado: superficies inclinadas suaves en vez de "escalones" planos por parche.
    Totalmente vectorizado (antes había un bucle con np.percentile por superpíxel: muy lento).
    """
    H, W = d.shape
    n_sp = int(slic_map.max()) + 1
    lab = slic_map.ravel()
    ys, xs = np.mgrid[0:H, 0:W]
    x = (xs / W).ravel().astype(np.float64)
    y = (ys / H).ravel().astype(np.float64)
    v = d.ravel().astype(np.float64)
    cnt = np.bincount(lab, minlength=n_sp).astype(np.float64) + 1e-9

    def S(a):
        return np.bincount(lab, weights=a, minlength=n_sp) / cnt

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
    plano = np.clip(plano, vmin[lab], vmax[lab])       # el plano nunca se sale del rango observado
    return plano.reshape(H, W).astype(np.float32)


def _prior_suelo(H: int, W: int, y_horizonte: float) -> np.ndarray:
    """
    Disparidad de un suelo plano visto en perspectiva.
    Con cámara a altura h: Z = f·h / (y - y_h)  =>  disparidad 1/Z ∝ (y - y_h): LINEAL en y.
    (La versión anterior usaba una potencia 1.10 sobre Z, que no corresponde a ninguna geometría.)
    """
    y = (np.arange(H, dtype=np.float32) / H).reshape(H, 1)
    g = np.clip((y - y_horizonte) / (1.0 - y_horizonte + 1e-6), 0.0, 1.0)
    return np.repeat(g, W, axis=1)


def _anclar_verticales_al_suelo(d, es_suelo, es_cielo, peso_vertical):
    """
    Un objeto vertical (tronco, poste, pared, persona) tiene la profundidad del punto donde
    TOCA EL SUELO. Para cada píxel buscamos el píxel de suelo más cercano hacia abajo en su
    columna y, si la red lo situó más cerca que esa base (copas que "se vienen encima"),
    se tira de él hacia la profundidad de la base. Es la idea raíz->copa de la versión original,
    pero con un barrido vectorizado en vez de dilataciones iterativas.
    """
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
    """Recta y ≈ a·x + b con descarte iterativo de atípicos (MAD)."""
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


# --------------------------------------------------------------------------------------
# Detección de cielo (común a ambos modos)
# --------------------------------------------------------------------------------------
def _mascara_cielo(img_lab, enfoque, y_norm, y_horizonte, disp=None):
    L = img_lab[:, :, 0] / 255.0
    b_c = img_lab[:, :, 2] - 128.0
    a_c = img_lab[:, :, 1] - 128.0
    color = (L > 0.52) & (b_c < 12.0) & (a_c > -12.0) & (enfoque < 0.38) & (y_norm < y_horizonte + 0.03)
    montana = ((y_norm < y_horizonte + 0.03) & (y_norm > y_horizonte - 0.12) &
               (L < 0.55) & (b_c < 8.0) & (enfoque < 0.25))
    cielo = color | montana
    if disp is not None:   # con red neuronal exigimos además que sea "lejano"
        cielo &= disp < np.percentile(disp, 35)
    cielo = cv2.morphologyEx(cielo.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # conserva solo regiones grandes (descarta "cielo" falso dentro de objetos)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(cielo, connectivity=8)
    grande = np.zeros(n, dtype=bool)
    grande[1:] = stats[1:, cv2.CC_STAT_AREA] > 0.01 * cielo.size
    return grande[comp]


# --------------------------------------------------------------------------------------
# Modo geométrico (heurística original mejorada + relieve volumétrico orgánico)
# --------------------------------------------------------------------------------------
def _profundidad_volumetrica_objeto(img_lab: np.ndarray, enfoque: np.ndarray, mascara_fg: np.ndarray) -> np.ndarray:
    """
    Genera un perfil de profundidad suave y convexo tipo cúpula/esfera para un objeto segmentado,
    modulando con el sombreado (luminancia), saturación y detalle local para que la superficie
    no tenga cortes de horizonte ni escalones artificiales.
    """
    m_u8 = mascara_fg.astype(np.uint8)
    dist = cv2.distanceTransform(m_u8, cv2.DIST_L2, 5).astype(np.float32)
    max_d = float(dist.max())
    if max_d < 1e-3:
        return np.zeros_like(enfoque, dtype=np.float32)

    # Perfil elipsoidal: z = sqrt(1 - (1 - r)^2) = sqrt(2r - r^2), suave hasta el borde
    r = np.clip(dist / max_d, 0.0, 1.0)
    domo = np.sqrt(np.clip(2.0 * r - r * r, 0.0, 1.0))
    domo = cv2.GaussianBlur(domo, (0, 0), sigmaX=max(2.0, max_d * 0.08))
    if domo[mascara_fg].max() > 1e-6:
        domo = domo / float(domo[mascara_fg].max())

    # Sombreado suave (Shape-from-Shading ligero)
    L = img_lab[:, :, 0] / 255.0
    L_suave = cv2.GaussianBlur(L.astype(np.float32), (9, 9), 0)
    # Descontar brillos especulares puntuales muy intensos para que no creen picos puntiagudos
    especular = np.clip((L - 0.85) / 0.15, 0.0, 1.0)
    sombreado = L_suave * (1.0 - 0.35 * especular) + 0.15 * enfoque

    # Combinar domo geométrico (82%) + relieve fotométrico (18%)
    z = np.zeros_like(domo, dtype=np.float32)
    z_interno = 0.25 + 0.62 * domo + 0.13 * sombreado
    z[mascara_fg] = np.clip(z_interno[mascara_fg], 0.15, 1.0)
    return z


def _profundidad_geometrica(img_lab, enfoque, y_norm, x_norm, y_horizonte, x_vp, es_cielo,
                            vp_confianza: float = 1.0, mascara_fg: np.ndarray | None = None):
    if mascara_fg is not None:
        return _profundidad_volumetrica_objeto(img_lab, enfoque, mascara_fg)

    H, W = y_norm.shape
    L = img_lab[:, :, 0] / 255.0
    a_c = img_lab[:, :, 1] - 128.0
    b_c = img_lab[:, :, 2] - 128.0

    # Si la confianza del punto de fuga es baja (ej. primer plano de un objeto, retrato o mesa),
    # combinamos una estimación de saliencia/objeto central con un gradiente de soporte suave,
    # evitando el corte duro en y_horizonte que partía los objetos por la mitad.
    if vp_confianza < 0.25:
        from fase1_slic import segmentar_objeto_sin_fondo
        img_bgr_aprox = cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
        fg_auto = segmentar_objeto_sin_fondo(img_bgr_aprox)
        ratio_fg = float(fg_auto.mean())
        if 0.04 < ratio_fg < 0.85:
            z_obj = _profundidad_volumetrica_objeto(img_lab, enfoque, fg_auto)
            # Fondo con gradiente vertical muy suave detrás del objeto
            z_bg = 0.05 + 0.22 * np.clip(y_norm, 0.0, 1.0)
            z_bg[es_cielo] = 0.0
            z = np.where(fg_auto, np.maximum(z_obj, z_bg + 0.12), z_bg)
            return cv2.GaussianBlur(z.astype(np.float32), (5, 5), 0)

    arriba_no_cielo = (y_norm < y_horizonte + 0.02) & (~es_cielo)
    gx = cv2.Sobel(L.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    borde_v = np.abs(gx) / (np.percentile(np.abs(gx), 98) + 1e-6)
    abajo_obj = (y_norm >= y_horizonte - 0.02) & (y_norm < 0.88) & (
        (L < 0.34) | ((enfoque > 0.32) & (borde_v > 0.28)) |
        ((a_c < -10.0) & (enfoque > 0.28) & (np.abs(x_norm - x_vp) > 0.12)))
    es_vertical = arriba_no_cielo | abajo_obj
    cerrado = cv2.morphologyEx(es_vertical.astype(np.uint8), cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (7, 15))) > 0
    es_vertical = cerrado & (~es_cielo)

    z_suelo = _prior_suelo(H, W, y_horizonte - 0.02)
    z_suelo = np.clip(z_suelo * 0.94 + 0.06, 0.06, 1.0)

    # Raíz -> copa, vectorizado: fila más baja de objeto vertical en cada columna
    base_valida = es_vertical & (y_norm >= y_horizonte - 0.01)
    hay = base_valida.any(axis=0)
    fila_base = H - 1 - np.argmax(base_valida[::-1, :], axis=0)
    z_raiz_col = np.where(hay, z_suelo[fila_base, np.arange(W)], 0.0).astype(np.float32)
    z_obj = np.where(es_vertical, z_raiz_col[None, :], 0.0).astype(np.float32)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    z_prop = z_obj.copy()
    for _ in range(18):
        z_dil = cv2.dilate(z_prop, kern)
        z_prop = np.where(es_vertical, np.maximum(z_prop, z_dil * 0.992), 0.0)

    z = z_suelo.copy()
    z[es_cielo] = 0.0
    relieve = 0.05 * L + 0.04 * enfoque
    z[es_vertical] = np.clip(z_prop[es_vertical] + relieve[es_vertical], 0.12, 1.0)
    return z.astype(np.float32)


# --------------------------------------------------------------------------------------
# Función principal (misma firma de retorno que tu versión original)
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
    img_lab = info_escena["lab_map"]
    enfoque = info_escena["enfoque_map"]

    # 1. K-Means sobre el hipervector (agrupa superpíxeles con apariencia similar)
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

    mascara_fg = info_escena.get("mascara_fg", None)
    usa_suelo = False
    if disp_neural is not None:
        modo = "neural"
        # 2a. Refinar bordes de la red con la imagen
        d = filtro_guiado_rapido(img_rgb, disp_neural, radio=4, eps=1e-3)
        d = np.clip(d, 0.0, 1.0)

        # 2b. Cielo al infinito
        es_cielo = _mascara_cielo(img_lab, enfoque, y_norm, y_h, d)
        d[es_cielo] = 0.0

        # 2c. Corrección con suelo plano en perspectiva (solo si hay perspectiva lineal fiable)
        if mascara_fg is None and vp_conf >= conf_min_perspectiva:
            g = _prior_suelo(H, W, y_h)
            zona = (y_norm > max(y_h + 0.10, 0.6)) & (~es_cielo)       # tercio inferior: casi seguro suelo
            a, b, _ = _ajuste_robusto(g[zona][::7], d[zona][::7])
            if a > 0.05:   # el suelo debe acercarse hacia abajo; si no, el ajuste no es creíble
                g_al = np.clip(a * g + b, 0.0, 1.0)
                es_suelo = (y_norm > y_h) & (np.abs(d - g_al) < 0.08) & (~es_cielo)
                es_suelo = cv2.morphologyEx(es_suelo.astype(np.uint8), cv2.MORPH_OPEN,
                                            np.ones((5, 5), np.uint8)) > 0
                d = np.where(es_suelo, d + peso_suelo * (g_al - d), d)
                d = _anclar_verticales_al_suelo(d, es_suelo, es_cielo, peso_vertical)
                usa_suelo = True
        if mascara_fg is not None:
            # Mezclar relieve neural con el perfil volumétrico del objeto para bordes limpios
            z_domo = _profundidad_volumetrica_objeto(img_lab, enfoque, mascara_fg)
            d = np.where(mascara_fg, 0.60 * d + 0.40 * z_domo, 0.0)
        mapa_geom = d
    else:
        modo = "geometrico"
        es_cielo = _mascara_cielo(img_lab, enfoque, y_norm, y_h) if mascara_fg is None else ~mascara_fg
        mapa_geom = _profundidad_geometrica(
            img_lab, enfoque, y_norm, x_norm, y_h, x_vp, es_cielo,
            vp_confianza=vp_conf, mascara_fg=mascara_fg
        )

    # 3. Regularización por superpíxeles: planos inclinados (Make3D) + mezcla con el píxel
    planos = _ajuste_planos_por_superpixel(mapa_geom, slic_map)
    mapa_z_bruto = 0.65 * mapa_geom + 0.35 * planos

    # 4. Filtro guiado para devolver los bordes a la silueta real de los objetos
    mapa_z_final = filtro_guiado_rapido(img_rgb, mapa_z_bruto, radio=6, eps=5e-4)
    if mascara_fg is not None:
        # Normalizar únicamente dentro del objeto y anular el fondo
        if mascara_fg.any():
            lo, hi = np.percentile(mapa_z_final[mascara_fg], [2, 99])
            z_fg = np.clip((mapa_z_final - lo) / (hi - lo + 1e-6), 0.0, 1.0)
            # Garantizar un borde suave que converja a 0.20 en el contorno para que el espejo cierre perfecto
            dist_borde = cv2.distanceTransform(mascara_fg.astype(np.uint8), cv2.DIST_L2, 5)
            peso_borde = np.clip(dist_borde / max(float(dist_borde.max()) * 0.22, 3.0), 0.0, 1.0)
            mapa_z_final = np.where(mascara_fg, 0.20 + 0.80 * z_fg * np.sqrt(peso_borde), 0.0).astype(np.float32)
        else:
            mapa_z_final = _normalizar(mapa_z_final)
    else:
        mapa_z_final = _normalizar(mapa_z_final)
        mapa_z_final[es_cielo] = np.minimum(mapa_z_final[es_cielo], 0.02)

    metricas = {
        "modo_profundidad": modo,
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
