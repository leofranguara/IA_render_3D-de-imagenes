"""
Fase 2: K-Means sobre superpíxeles + Motor Universal de Profundidad Invariante al Albedo (100% Offline).

Principios clave que garantizan modelos 3D casi perfectos en cualquier fotografía:
  1. Separación Intrínseca Albedo vs. Geometría (Invarianza al Color Oscuro/Claro):
     - Evita el error clásico donde zonas oscuras (barba negra, cabello, cejas, vidrios polarizados
       de autos, llantas negras, ventanas oscuras de casas) se hunden como agujeros, o donde zonas
       claras (frente iluminada, cuello, camisas blancas, paredes blancas) se inflan artificialmente.
     - El relieve fotométrico se extrae mediante un filtro paso-banda intrínseco con media cero
       dentro de cada región de material.
  2. Modelo Anatómico 3D para Retratos y Personas (`_profundidad_anatomica_retrato`):
     - Reconstruye el elipsoide craneal unificado (cabello + frente + rostro + barba/mentón en una
       misma superficie 3D continua), escultura facial 3D (puente y punta de la nariz, arco ciliar,
       pómulos, labios, prominencia del mentón, cuencas oculares suaves y retroceso de orejas),
       cuello cilíndrico situado detrás de la mandíbula y busto/hombros con curvatura controlada.
  3. Modelo Volumétrico Estructural para Objetos y Vehículos (`_profundidad_volumetrica_objeto`):
     - Combina transformada de distancia global con distancia normalizada al eje medial (esqueleto)
       y biselado en los 4 bordes del marco para que ninguna parte ancha se infle desproporcionadamente.
  4. Modelo Multi-Pista por Capas para Paisajes, Casas y Escenas Completas (`_profundidad_escena_universal`):
     - Combina plano de suelo en perspectiva, transmisión atmosférica (Dark Channel + contraste),
       nitidez multi-escala y extrusión vertical raíz->techo invariante al albedo para fachadas y árboles.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score


# --------------------------------------------------------------------------------------
# Utilidades de filtrado e invarianza al albedo
# --------------------------------------------------------------------------------------
def filtro_guiado_rapido(guia_rgb: np.ndarray, mapa_z: np.ndarray, radio: int = 4,
                          eps: float = 1e-3) -> np.ndarray:
    """Filtro guiado ligero para alinear bordes sin transferir el brillo del albedo a la profundidad."""
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


def _sombreado_intrinseco_invariante_albedo(img_lab: np.ndarray, amplitud: float = 0.05) -> np.ndarray:
    """
    Extrae el micro-relieve geométrico de la superficie eliminando los cambios de color/albedo.
    Una barba negra, cabello oscuro, vidrio polarizado o llanta negra tiene media local igual a su
    entorno inmediato tras el filtro paso-banda acotado por tanh, por lo que NO se hunde en Z.
    """
    L = img_lab[:, :, 0].astype(np.float32) / 255.0
    L_fino = cv2.GaussianBlur(L, (5, 5), 1.0)
    L_medio = cv2.GaussianBlur(L, (21, 21), 5.0)
    delta = L_fino - L_medio

    # Atenuar en bordes de cambio brusco de material (albedo edges: ej. piel -> barba o carrocería -> ventana)
    gx = cv2.Sobel(L_fino, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L_fino, cv2.CV_32F, 0, 1, ksize=3)
    grad_l = cv2.GaussianBlur(np.hypot(gx, gy), (9, 9), 0)
    peso_continuo = 1.0 / (1.0 + (grad_l / 0.12) ** 2)

    relieve = amplitud * np.tanh((delta * peso_continuo) / 0.06)
    return cv2.GaussianBlur(relieve.astype(np.float32), (5, 5), 0)


def _distancia_con_bordes_cerrados(mascara_fg: np.ndarray) -> np.ndarray:
    """
    Calcula la Transformada de Distancia Euclidiana asegurando que si el objeto toca cualquiera
    de los 4 bordes de la fotografía (como el pecho/hombros en un retrato), el borde del marco
    actúe como frontera con un bisel suave, evitando que la base se infle como un globo.
    """
    H, W = mascara_fg.shape
    m_pad = np.zeros((H + 2, W + 2), dtype=np.uint8)
    m_pad[1:-1, 1:-1] = mascara_fg.astype(np.uint8)
    dist_cerrada = cv2.distanceTransform(m_pad, cv2.DIST_L2, 5)[1:-1, 1:-1].astype(np.float32)

    # Transformada abierta (solo respecto al fondo recortado) para combinar en bordes que tocan el marco
    dist_abierta = cv2.distanceTransform(mascara_fg.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    # 75% cerrada + 25% abierta: evita abombamiento en el borde inferior pero mantiene cierre suave
    return 0.75 * dist_cerrada + 0.25 * dist_abierta


def _ajuste_planos_por_superpixel(d: np.ndarray, slic_map: np.ndarray,
                                  ridge: float = 5e-5, max_pend: float = 1.8) -> np.ndarray:
    """
    Ajusta un plano d ≈ c + a·(x-mx) + b·(y-my) dentro de cada superpíxel (Make3D).
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
    cielo = (L > 0.54) & (b_c < 10.0) & (a_c > -12.0) & (enfoque < 0.30) & (y_norm < y_horizonte + 0.06)
    if disp is not None:
        cielo &= disp < np.percentile(disp, 35)
    cielo = cv2.morphologyEx(cielo.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, comp, stats, _ = cv2.connectedComponentsWithStats(cielo, connectivity=8)
    grande = np.zeros(n, dtype=bool)
    grande[1:] = stats[1:, cv2.CC_STAT_AREA] > 0.01 * cielo.size
    return grande[comp]


def _pista_nitidez_multiescala(img_lab: np.ndarray, enfoque: np.ndarray, grad_map: np.ndarray) -> np.ndarray:
    L = img_lab[:, :, 0].astype(np.float32) / 255.0
    dif_fina = np.abs(L - cv2.GaussianBlur(L, (5, 5), 1.0))
    dif_media = np.abs(L - cv2.GaussianBlur(L, (15, 15), 3.0))
    dif_gruesa = np.abs(L - cv2.GaussianBlur(L, (31, 31), 7.0))

    energia = (
        0.40 * cv2.GaussianBlur(dif_fina, (9, 9), 0) / (np.percentile(dif_fina, 98) + 1e-6) +
        0.35 * cv2.GaussianBlur(dif_media, (15, 15), 0) / (np.percentile(dif_media, 98) + 1e-6) +
        0.25 * cv2.GaussianBlur(dif_gruesa, (21, 21), 0) / (np.percentile(dif_gruesa, 98) + 1e-6)
    )
    combinada = 0.55 * np.clip(energia, 0.0, 1.0) + 0.25 * enfoque + 0.20 * cv2.GaussianBlur(grad_map, (9, 9), 0)
    return _normalizar(combinada, 2, 98)


def _pista_saliencia_superpixeles(slic_map: np.ndarray, hipervectores: np.ndarray) -> np.ndarray:
    n_sp = len(hipervectores)
    sp_x, sp_y = hipervectores[:, 0], hipervectores[:, 1]
    sp_color = hipervectores[:, 2:5]

    en_marco = (sp_y < 0.12) | (sp_x < 0.07) | (sp_x > 0.93)
    if en_marco.sum() < 5:
        en_marco = np.ones(n_sp, dtype=bool)
    bg_color = sp_color[en_marco]

    diff = sp_color[:, None, :] - bg_color[None, :, :]
    dist_c = np.linalg.norm(diff, axis=2)
    k_min = min(5, dist_c.shape[1])
    contraste_bg = np.partition(dist_c, k_min - 1, axis=1)[:, :k_min].mean(axis=1)
    contraste_bg = (contraste_bg - contraste_bg.min()) / (contraste_bg.max() - contraste_bg.min() + 1e-6)

    peso_centro = np.exp(-((sp_x - 0.5) ** 2 / 0.32 + (sp_y - 0.56) ** 2 / 0.35))
    sal_sp = np.clip(0.65 * contraste_bg + 0.35 * peso_centro, 0.0, 1.0).astype(np.float32)
    return cv2.GaussianBlur(sal_sp[slic_map], (11, 11), 0)


# --------------------------------------------------------------------------------------
# 1A. Modelo Anatómico 3D para Retratos, Rostros y Personas
# --------------------------------------------------------------------------------------
def _profundidad_anatomica_retrato(
    img_rgb: np.ndarray,
    img_lab: np.ndarray,
    mascara_fg: np.ndarray,
    anat: dict,
) -> np.ndarray:
    """
    Reconstruye la geometría 3D anatómica de una cabeza, rostro, cuello y busto humano:
      - Elipsoide craneal unificado: el cabello oscuro, la frente, las mejillas y la barba negra
        forman parte de un mismo volumen craneal continuo (la barba y las cejas NUNCA se hunden).
      - Escultura facial 3D: puente y punta de la nariz sobresalientes, volumen de labios y mentón,
        arco ciliar y pómulos suaves, cuencas oculares anatómicas y orejas retraídas lateralmente.
      - Cuello cilíndrico situado DETRÁS del mentón y la mandíbula (elimina el abombamiento del cuello).
      - Hombros y torso con curvatura cilíndrica suave y controlada.
    """
    H, W = mascara_fg.shape
    y_top = anat["y_top"]
    y_chin = anat["y_chin"]
    x_mid = anat["x_mid"]
    r_x = max(anat["r_x"], 15.0)
    r_y = max(anat["r_y"], 15.0)

    ancho_suave = cv2.GaussianBlur(anat["ancho_fila"].reshape(H, 1), (1, 31), 0)
    centro_suave = cv2.GaussianBlur(anat["centro_fila"].reshape(H, 1), (1, 31), 0)

    grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)

    # Coordenadas normalizadas respecto al óvalo de la cabeza
    # u_h en [-1, 1] de oreja a oreja; v_h en [0, 1] desde la coronilla (0.0) hasta la punta del mentón (1.0)
    u_h = (grid_x - x_mid) / (r_x + 1e-6)
    v_h = (grid_y - y_top) / (2.0 * r_y + 1e-6)

    # Curva parabólica de la mandíbula: en el centro (u_h=0) llega hasta v_h=1.0 (mentón),
    # hacia los lados (|u_h|->0.85) sube suavemente hacia la base de las orejas (v_h≈0.74)
    v_jaw = 1.0 - 0.28 * np.minimum(u_h ** 2, 1.2)

    # Máscara suave de pertenencia a la cabeza (vs. cuello y torso debajo de la mandíbula)
    alpha_head = (
        1.0 / (1.0 + np.exp((v_h - v_jaw) / 0.042)) *
        1.0 / (1.0 + np.exp((np.abs(u_h) - 1.05) / 0.07))
    ).astype(np.float32)

    # 1. Elipsoide craneal base (continuo sobre cabello, frente, rostro y barba)
    semi_ancho_fila = np.maximum(0.5 * ancho_suave, 8.0)
    u_row = (grid_x - centro_suave) / semi_ancho_fila
    # Coordenada vertical craneal centrada en la mitad del rostro (v_h = 0.54)
    v_c = (v_h - 0.53) / 0.53
    z_craneo = np.sqrt(np.clip(1.0 - 0.76 * (u_row ** 2) - 0.50 * (v_c ** 2), 0.0, 1.0))

    # 2. Rasgos anatómicos 3D del rostro
    # Puente y punta de la nariz (sobresale en el eje central entre v_h=0.40 y v_h=0.65)
    dz_nariz = (
        0.18 * np.exp(-(u_h ** 2) / (2.0 * 0.125 ** 2) - ((v_h - 0.54) ** 2) / (2.0 * 0.105 ** 2)) +
        0.09 * np.exp(-(u_h ** 2) / (2.0 * 0.090 ** 2) - ((v_h - 0.60) ** 2) / (2.0 * 0.048 ** 2))
    )
    # Labios / boca (v_h ≈ 0.74)
    dz_boca = 0.075 * np.exp(-(u_h ** 2) / (2.0 * 0.18 ** 2) - ((v_h - 0.74) ** 2) / (2.0 * 0.045 ** 2))
    # Prominencia del mentón / barbilla (v_h ≈ 0.89, sobresale hacia adelante aunque tenga barba oscura)
    dz_menton = 0.115 * np.exp(-(u_h ** 2) / (2.0 * 0.23 ** 2) - ((v_h - 0.89) ** 2) / (2.0 * 0.068 ** 2))
    # Pómulos y arco superciliar (cejas/frente inferior)
    dz_pomulos = (
        0.055 * np.exp(-((np.abs(u_h) - 0.36) ** 2) / (2.0 * 0.16 ** 2) - ((v_h - 0.55) ** 2) / (2.0 * 0.09 ** 2)) +
        0.045 * np.exp(-(u_h ** 2) / (2.0 * 0.44 ** 2) - ((v_h - 0.36) ** 2) / (2.0 * 0.055 ** 2))
    )
    # Cuencas oculares suaves (v_h ≈ 0.45, |u_h| ≈ 0.33)
    dz_orbitas = -0.042 * np.exp(-((np.abs(u_h) - 0.33) ** 2) / (2.0 * 0.12 ** 2) - ((v_h - 0.45) ** 2) / (2.0 * 0.040 ** 2))
    # Orejas (situadas en los extremos laterales |u_h| > 0.76 entre v_h=0.40 y 0.68, más atrás que las mejillas)
    peso_oreja = np.clip((np.abs(u_h) - 0.76) / 0.24, 0.0, 1.0)
    dz_orejas = -0.14 * peso_oreja * np.exp(-((v_h - 0.53) ** 2) / (2.0 * 0.13 ** 2))

    z_head = 0.76 * z_craneo + dz_nariz + dz_boca + dz_menton + dz_pomulos + dz_orbitas + dz_orejas

    # 3. Cuello cilíndrico (situado DETRÁS del mentón/mandíbula) y busto/hombros
    u_cuello = (grid_x - x_mid) / (0.62 * r_x + 1e-6)
    z_cuello = 0.50 * np.sqrt(np.clip(1.0 - 0.72 * (u_cuello ** 2), 0.0, 1.0))

    # Torso / hombros: curvatura cilíndrica horizontal suave con altura máxima controlada (0.54)
    # para que el pecho nunca sobresalga más que la cara
    z_torso = 0.54 * np.sqrt(np.clip(1.0 - 0.82 * (u_row ** 2), 0.0, 1.0))
    peso_torso = np.clip((v_h - 1.08) / 0.22, 0.0, 1.0)
    z_body = (1.0 - peso_torso) * z_cuello + peso_torso * np.maximum(z_torso, z_cuello * 0.9)

    # Fusión anatómica Cabeza + Cuello/Torso
    z_anat = alpha_head * z_head + (1.0 - alpha_head) * z_body
    z_anat = cv2.GaussianBlur(z_anat.astype(np.float32), (9, 9), 0)

    # 4. Micro-relieve intrínseco invariante al albedo (añade detalle fino de labios, ojos, pelo y ropa sin hundir la barba)
    micro_relieve = _sombreado_intrinseco_invariante_albedo(img_lab, amplitud=0.04)

    # 5. Envolvente suave de borde para que el contorno cierre perfectamente en el espejo
    dist_borde = _distancia_con_bordes_cerrados(mascara_fg)
    max_db = max(float(np.percentile(dist_borde[mascara_fg], 95)), 1.0)
    env_borde = np.sqrt(np.clip(dist_borde / (max_db * 0.25 + 1e-6), 0.0, 1.0))

    z_total = (z_anat + micro_relieve) * env_borde
    z = np.zeros((H, W), dtype=np.float32)
    if mascara_fg.any():
        vals = z_total[mascara_fg]
        lo, hi = np.percentile(vals, [1, 99.5])
        z_norm = np.clip((z_total - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        z[mascara_fg] = np.clip(0.14 + 0.86 * z_norm[mascara_fg], 0.14, 1.0)
    return z


# --------------------------------------------------------------------------------------
# 1B. Modelo Volumétrico Invariante al Albedo para Objetos, Vehículos, Frutas y Animales
# --------------------------------------------------------------------------------------
def _perfil_revolucion_objeto(mascara_fg: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Calcula un perfil volumétrico cilíndrico/de revolución por fila horizontal a partir de la
    silueta del objeto y estima su grado de simetría izquierda-derecha (IoU).
    Ideal para jarrones, botellas, tazas, frutas y objetos axialmente simétricos.
    """
    H, W = mascara_fg.shape
    ys_fg, xs_fg = np.where(mascara_fg)
    if len(ys_fg) < 20:
        return np.zeros((H, W), dtype=np.float32), 0.0

    y_min, y_max = int(ys_fg.min()), int(ys_fg.max())
    x_min, x_max = int(xs_fg.min()), int(xs_fg.max())

    # Simetría horizontal (IoU con el reflejo izquierda-derecha del recorte)
    recorte = mascara_fg[y_min:y_max + 1, x_min:x_max + 1]
    inter = float((recorte & recorte[:, ::-1]).sum())
    union = float((recorte | recorte[:, ::-1]).sum()) + 1e-6
    simetria_lr = inter / union

    hay_fila = mascara_fg.any(axis=1)
    filas_validas = np.where(hay_fila)[0]
    x_izq = np.argmax(mascara_fg, axis=1).astype(np.float32)
    x_der = (W - 1 - np.argmax(mascara_fg[:, ::-1], axis=1)).astype(np.float32)

    # Interpolar y suavizar los perfiles de ancho y centro para evitar bandas horizontales
    x_izq_s = np.interp(np.arange(H), filas_validas, x_izq[filas_validas])
    x_der_s = np.interp(np.arange(H), filas_validas, x_der[filas_validas])
    k_suave = max(5, (int(0.05 * (y_max - y_min + 1)) | 1))
    x_izq_s = cv2.GaussianBlur(x_izq_s.reshape(H, 1), (1, k_suave), 0).ravel()
    x_der_s = cv2.GaussianBlur(x_der_s.reshape(H, 1), (1, k_suave), 0).ravel()

    radio_fila = np.maximum(0.5 * (x_der_s - x_izq_s), 1.0)
    centro_fila = 0.5 * (x_izq_s + x_der_s)
    r_max = max(float(radio_fila[filas_validas].max()), 1.0)

    xs = np.arange(W, dtype=np.float32)[None, :]
    u_rel = (xs - centro_fila[:, None]) / radio_fila[:, None]
    perfil = (radio_fila[:, None] / r_max) * np.sqrt(np.clip(1.0 - u_rel * u_rel, 0.0, 1.0))
    perfil[~mascara_fg] = 0.0
    perfil = cv2.GaussianBlur(perfil.astype(np.float32), (5, 5), 0)
    if perfil[mascara_fg].max() > 1e-6:
        perfil = np.where(mascara_fg, perfil / float(perfil[mascara_fg].max()), 0.0).astype(np.float32)
    return perfil, simetria_lr


def _detectar_cavidad_superior(
    img_lab: np.ndarray,
    grad_map: np.ndarray,
    mascara_fg: np.ndarray,
) -> np.ndarray:
    """
    Detecta de forma no supervisada una abertura/cavidad cóncava en la parte superior de
    recipientes (jarrones, tazas, cuencos, macetas) buscando un arco de labio frontal que
    delimita una región elíptica interior más sombreada debajo del borde superior.
    Devuelve un mapa en [0, 1] con la profundidad relativa del hueco (0 si no hay cavidad).
    """
    H, W = mascara_fg.shape
    vacio = np.zeros((H, W), dtype=np.float32)
    ys_fg, xs_fg = np.where(mascara_fg)
    if len(ys_fg) < 100:
        return vacio

    y_min, y_max = int(ys_fg.min()), int(ys_fg.max())
    x_min, x_max = int(xs_fg.min()), int(xs_fg.max())
    alto_obj = y_max - y_min + 1
    ancho_obj = x_max - x_min + 1
    if alto_obj < 40 or ancho_obj < 40:
        return vacio

    # Verificar simetría horizontal del objeto (recipientes como jarrones, tazas y cuencos son simétricos)
    recorte = mascara_fg[y_min:y_max + 1, x_min:x_max + 1]
    simetria_lr = float((recorte & recorte[:, ::-1]).sum()) / (float((recorte | recorte[:, ::-1]).sum()) + 1e-6)
    if simetria_lr < 0.72:
        return vacio

    # Verificar que exista una boca superior ancha (no una punta afilada o tallo)
    y_probe = min(H - 1, y_min + max(3, int(0.06 * alto_obj)))
    xs_boca = np.where(mascara_fg[y_probe])[0]
    if len(xs_boca) < 0.28 * ancho_obj:
        return vacio

    x_b_min, x_b_max = int(xs_boca.min()), int(xs_boca.max())
    ancho_boca = x_b_max - x_b_min + 1
    cx_boca = 0.5 * (x_b_min + x_b_max)
    # La boca debe estar razonablemente centrada respecto al objeto
    if abs(cx_boca - 0.5 * (x_min + x_max)) > 0.16 * ancho_obj:
        return vacio

    L = cv2.GaussianBlur(img_lab[:, :, 0].astype(np.float32) / 255.0, (5, 5), 0)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    # El labio frontal genera una transición vertical fuerte (interior sombreado -> borde iluminado)
    borde_lip = np.maximum(gy, 0.0) + 0.45 * np.abs(gy)
    borde_lip = cv2.GaussianBlur(borde_lip, (11, 3), 0)

    margen_x = max(3, int(0.10 * ancho_boca))
    c_ini = x_b_min + margen_x
    c_fin = x_b_max - margen_x
    if c_fin - c_ini < 12:
        return vacio

    y_banda_max = min(H - 1, y_min + int(0.22 * alto_obj))
    min_prof_px = max(3, int(0.03 * alto_obj))
    max_prof_px = max(min_prof_px + 4, int(0.18 * alto_obj))

    cols_det = []
    y_top_det = []
    y_lip_det = []
    fuerza_det = []

    for x in range(c_ini, c_fin + 1):
        col_fg = np.where(mascara_fg[y_min:y_banda_max + 1, x])[0]
        if len(col_fg) < min_prof_px + 2:
            continue
        yt = y_min + int(col_fg[0])
        s0 = min(H - 1, yt + min_prof_px)
        s1 = min(y_banda_max + 1, yt + max_prof_px)
        if s1 <= s0 + 2:
            continue
        tramo = borde_lip[s0:s1, x]
        idx_max = int(np.argmax(tramo))
        val_max = float(tramo[idx_max])
        if val_max >= 0.035:
            cols_det.append(x)
            y_top_det.append(yt)
            y_lip_det.append(s0 + idx_max)
            fuerza_det.append(val_max)

    if len(cols_det) < max(10, int(0.45 * (c_fin - c_ini + 1))):
        return vacio

    cols_arr = np.array(cols_det, dtype=np.float32)
    y_top_arr = np.array(y_top_det, dtype=np.float32)
    y_lip_arr = np.array(y_lip_det, dtype=np.float32)
    w_arr = np.array(fuerza_det, dtype=np.float32)

    # Ajustar una parábola suave al arco del labio frontal (y_lip en función de x)
    xn = (cols_arr - cx_boca) / max(0.5 * ancho_boca, 1.0)
    coef = np.polyfit(xn, y_lip_arr, 2, w=w_arr)
    # Descartar si el arco se curva bruscamente hacia arriba (no corresponde a una elipse vista desde arriba)
    if coef[0] > 0.12 * alto_obj:
        return vacio

    # Reconstruir la curva del labio y del borde superior en todo el ancho de la boca
    xs_all = np.arange(x_b_min, x_b_max + 1, dtype=np.float32)
    xn_all = (xs_all - cx_boca) / max(0.5 * ancho_boca, 1.0)
    y_lip_curva = np.polyval(coef, xn_all)

    y_top_curva = np.full_like(xs_all, float(y_min))
    for i, x in enumerate(xs_all.astype(int)):
        col_fg = np.where(mascara_fg[y_min:y_banda_max + 1, x])[0]
        if len(col_fg) > 0:
            y_top_curva[i] = float(y_min + int(col_fg[0]))
    y_top_curva = cv2.GaussianBlur(y_top_curva.reshape(1, -1), (9, 1), 0).ravel()

    # El borde superior de una boca/abertura es casi horizontal (no una silueta inclinada)
    if float(np.std(y_top_curva)) > 0.04 * alto_obj:
        return vacio

    alturas = y_lip_curva - y_top_curva
    h_med = float(np.median(alturas))
    if h_med < 0.03 * alto_obj or h_med > 0.19 * alto_obj:
        return vacio
    # Una abertura circular en perspectiva es una elipse horizontal más ancha que alta
    if ancho_boca < 1.8 * h_med:
        return vacio

    # Validar contraste fotométrico: el interior de la boca debe estar más sombreado/oscuro que el labio
    mascara_int = np.zeros((H, W), dtype=bool)
    mascara_labio = np.zeros((H, W), dtype=bool)
    for i, x in enumerate(xs_all.astype(int)):
        if x < c_ini or x > c_fin:
            continue
        yt = int(round(y_top_curva[i])) + 2
        yl = int(round(y_lip_curva[i]))
        if yl > yt + 1:
            mascara_int[yt:yl, x] = True
            l0 = max(0, yl - 1)
            l1 = min(H, yl + max(3, int(0.025 * alto_obj)))
            mascara_labio[l0:l1, x] = True

    mascara_int &= mascara_fg
    mascara_labio &= mascara_fg
    if mascara_int.sum() < 25 or mascara_labio.sum() < 15:
        return vacio

    L_int = float(np.median(L[mascara_int]))
    L_labio = float(np.median(L[mascara_labio]))
    fuerza_media = float(np.mean(w_arr))
    # Evitar falsos positivos en zonas muy claras (reflejos blancos) o sin borde de labio real
    if L_int > 0.78 or fuerza_media < 0.055:
        return vacio
    if L_int > L_labio + 0.03:
        return vacio

    # Construir mapa suave de profundidad cóncava [0, 1] dentro de la elipse de la boca
    mapa_cav = np.zeros((H, W), dtype=np.float32)
    rx = max(0.5 * ancho_boca * 0.90, 1.0)
    for i, x in enumerate(xs_all.astype(int)):
        tx = (float(x) - cx_boca) / rx
        if abs(tx) >= 1.0:
            continue
        yt = y_top_curva[i] + 1.5
        yl = y_lip_curva[i]
        span = yl - yt
        if span <= 2.0:
            continue
        y0 = max(0, int(np.floor(yt)))
        y1 = min(H, int(np.ceil(yl)))
        ys = np.arange(y0, y1, dtype=np.float32)
        ty = np.clip((ys - yt) / span, 0.0, 1.0)
        # Perfil cóncavo suave: 0 en los bordes del labio y máximo hacia el interior central
        prof_col = np.sin(np.pi * ty) * np.sqrt(max(0.0, 1.0 - tx * tx))
        mapa_cav[y0:y1, x] = np.maximum(mapa_cav[y0:y1, x], prof_col)

    mapa_cav[~mascara_fg] = 0.0
    mapa_cav = cv2.GaussianBlur(mapa_cav, (7, 7), 0)
    if mapa_cav.max() > 1e-6:
        mapa_cav = np.clip(mapa_cav / float(mapa_cav.max()), 0.0, 1.0)
    return mapa_cav.astype(np.float32)


def _detectar_boquilla_frontal(
    img_lab: np.ndarray,
    grad_map: np.ndarray,
    mascara_fg: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Detecta de forma no supervisada una boca/boquilla cilíndrica frontal que sobresale hacia
    la cámara (ej. toma de aire de un turbo, lente, tubería o conducto):
      - Un interior oscuro compacto y casi circular/elíptico (el conducto hueco).
      - Rodeado por un collarín/anillo más luminoso dentro del objeto (el borde que sobresale).
    Devuelve (mapa_realce_anillo en [0, 1], mapa_cavidad_interior en [0, 1]) o (None, None).
    """
    H, W = mascara_fg.shape
    area_fg = int(mascara_fg.sum())
    if area_fg < 400:
        return None, None

    L = cv2.GaussianBlur(img_lab[:, :, 0].astype(np.float32) / 255.0, (5, 5), 0)
    vals_L = L[mascara_fg]
    # Regiones muy oscuras dentro del objeto (posible interior de conducto)
    umbral_osc = min(0.32, float(np.percentile(vals_L, 26)))
    oscuro = mascara_fg & (L <= umbral_osc)
    oscuro = cv2.morphologyEx(oscuro.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    oscuro = cv2.morphologyEx(oscuro.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0

    contornos, _ = cv2.findContours(oscuro.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None, None

    ys_grid, xs_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    mapa_realce = np.zeros((H, W), dtype=np.float32)
    mapa_cav = np.zeros((H, W), dtype=np.float32)
    detectado = False

    for cnt in contornos:
        area = float(cv2.contourArea(cnt))
        frac = area / float(area_fg)
        if frac < 0.012 or frac > 0.16:
            continue
        perim = float(cv2.arcLength(cnt, True))
        if perim < 1e-3:
            continue
        circularidad = 4.0 * np.pi * area / (perim * perim)
        if circularidad < 0.52:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspecto = max(bw, bh) / max(min(bw, bh), 1.0)
        if aspecto > 1.75:
            continue

        mask_hueco = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask_hueco, [cnt], -1, 1, thickness=-1)
        hueco_b = mask_hueco > 0

        r_eq = float(np.sqrt(area / np.pi))
        k_anillo = max(7, int(round(r_eq * 0.90)) | 1)
        anillo = (cv2.dilate(mask_hueco, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_anillo, k_anillo))) > 0) & (~hueco_b)
        anillo_fg = anillo & mascara_fg
        # El collarín debe rodear mayoritariamente al hueco dentro del objeto (descarta esquinas o ruedas)
        if anillo_fg.sum() < 0.65 * max(anillo.sum(), 1):
            continue

        L_hueco = float(np.median(L[hueco_b]))
        L_anillo = float(np.median(L[anillo_fg]))
        grad_borde = float(np.mean(grad_map[anillo_fg]))
        if L_anillo < 0.48 or (L_anillo - L_hueco) < 0.22 or grad_borde < 0.10:
            continue

        # Boquilla frontal confirmada: elevar el collarín hacia afuera y hundir el conducto interior
        detectado = True
        cx = x + 0.5 * bw
        cy = y + 0.5 * bh
        rx = max(0.5 * bw, 4.0)
        ry = max(0.5 * bh, 4.0)
        r_norm = np.hypot((xs_grid - cx) / rx, (ys_grid - cy) / ry)

        # Realce suave del tubo/collarín que sobresale (máximo sobre el anillo en r_norm ≈ 1.0..1.5)
        realce_local = np.exp(-0.5 * ((np.maximum(r_norm - 1.15, 0.0)) / 0.65) ** 2)
        realce_local = np.where(r_norm < 0.92, realce_local * np.clip(r_norm / 0.92, 0.0, 1.0), realce_local)
        mapa_realce = np.maximum(mapa_realce, realce_local.astype(np.float32))

        # Cavidad profunda dentro del orificio (máxima en el centro del hueco)
        cav_local = np.sqrt(np.clip(1.0 - (r_norm / 0.98) ** 2, 0.0, 1.0))
        cav_local = np.where(hueco_b, cav_local, 0.0)
        mapa_cav = np.maximum(mapa_cav, cav_local.astype(np.float32))

    if not detectado:
        return None, None

    mapa_realce = cv2.GaussianBlur(np.where(mascara_fg, mapa_realce, 0.0).astype(np.float32), (9, 9), 0)
    mapa_cav = cv2.GaussianBlur(np.where(mascara_fg, mapa_cav, 0.0).astype(np.float32), (5, 5), 0)
    if mapa_realce.max() > 1e-6:
        mapa_realce = np.clip(mapa_realce / float(mapa_realce.max()), 0.0, 1.0)
    if mapa_cav.max() > 1e-6:
        mapa_cav = np.clip(mapa_cav / float(mapa_cav.max()), 0.0, 1.0)
    return mapa_realce, mapa_cav


def _detectar_objeto_plano_extruido(
    img_lab: np.ndarray,
    grad_map: np.ndarray,
    mascara_fg: np.ndarray,
    mapa_clusters: np.ndarray,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """
    Detecta de forma 100% no supervisada (usando los clústeres de K-Means + planitud de luminancia
    + presencia de agujeros pasantes) si el objeto es una pieza mecánica / rígida de caras planas
    extruidas (ej. soportes, placas, engranajes) en lugar de un cuerpo orgánico/esférico.

    Devuelve (factor_plano en [0, 1], perfil_meseta o None, mapa_agujeros_ciegos o None).
    """
    area_fg = int(mascara_fg.sum())
    if area_fg < 200:
        return 0.0, None, None

    L = img_lab[:, :, 0].astype(np.float32) / 255.0
    clusters_fg = mapa_clusters[mascara_fg]
    ids, conteos = np.unique(clusters_fg, return_counts=True)
    idx_dom = int(np.argmax(conteos))
    c_dom = int(ids[idx_dom])
    frac_dom = float(conteos[idx_dom]) / float(area_fg)

    mask_dom = mascara_fg & (mapa_clusters == c_dom)
    std_L_dom = float(np.std(L[mask_dom]))
    grad_dom = float(np.median(grad_map[mask_dom]))
    L_dom = float(np.median(L[mask_dom]))

    # Verificar si la máscara contiene agujeros pasantes reales (fuerte indicador de pieza rígida/mecánica)
    huecos_pasantes = ndi.binary_fill_holes(mascara_fg) & (~mascara_fg)
    tiene_huecos = int(huecos_pasantes.sum()) > max(40, int(0.005 * area_fg))

    es_plano = (
        (frac_dom >= 0.45 and std_L_dom <= 0.075 and grad_dom <= 0.085) or
        (tiene_huecos and frac_dom >= 0.38 and std_L_dom <= 0.095 and grad_dom <= 0.11)
    )
    if not es_plano:
        return 0.0, None, None

    # Calcular factor de planitud suave en [0.70, 1.0]
    score_planitud = np.clip((0.095 - std_L_dom) / 0.06, 0.0, 1.0)
    factor_plano = float(np.clip(0.72 + 0.28 * score_planitud + (0.12 if tiene_huecos else 0.0), 0.70, 1.0))

    # Construir perfil de meseta plana con bisel corto en los bordes (sin abombar el centro)
    dist = _distancia_con_bordes_cerrados(mascara_fg)
    max_d = max(float(dist.max()), 1.0)
    r_bisel = np.clip(dist / max(max_d * 0.14, 4.0), 0.0, 1.0)
    meseta = np.sqrt(np.clip(2.0 * r_bisel - r_bisel * r_bisel, 0.0, 1.0)).astype(np.float32)

    # Identificar pared lateral extruida y agujeros ciegos (ej. perforaciones para tornillos)
    borde_ext = mascara_fg & (~cv2.erode(mascara_fg.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool))
    z_plano = meseta.copy()
    mapa_huecos_ciegos = np.zeros_like(L, dtype=np.float32)

    for cid, cnt_c in zip(ids, conteos):
        cid = int(cid)
        if cid == c_dom:
            continue
        frac_c = float(cnt_c) / float(area_fg)
        mask_c = mascara_fg & (mapa_clusters == cid)
        L_c = float(np.median(L[mask_c]))
        toca_borde_ext = float((mask_c & borde_ext).sum()) / float(cnt_c + 1e-6)

        if toca_borde_ext < 0.08 and L_c < L_dom - 0.05 and frac_c < 0.16:
            # Clúster interior oscuro (agujero ciego o tornillo): hundir suavemente
            d_c = cv2.distanceTransform(mask_c.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
            if d_c.max() > 1e-3:
                d_c /= float(d_c.max())
            z_plano = np.where(mask_c, np.maximum(meseta * 0.65 - 0.22 * d_c, 0.18), z_plano)
            mapa_huecos_ciegos = np.maximum(mapa_huecos_ciegos, 0.55 * d_c)
        elif toca_borde_ext >= 0.12 and frac_c < 0.35:
            # Pared lateral extruida en perspectiva: escalón ligeramente más bajo que la cara superior
            z_plano = np.where(mask_c, meseta * 0.78, z_plano)

    z_plano[~mascara_fg] = 0.0
    z_plano = cv2.GaussianBlur(z_plano.astype(np.float32), (3, 3), 0)
    if z_plano[mascara_fg].max() > 1e-6:
        z_plano = np.where(mascara_fg, z_plano / float(z_plano[mascara_fg].max()), 0.0).astype(np.float32)

    if mapa_huecos_ciegos.max() > 0.10:
        mapa_huecos_ciegos = cv2.GaussianBlur(mapa_huecos_ciegos, (5, 5), 0)
    else:
        mapa_huecos_ciegos = None

    return factor_plano, z_plano, mapa_huecos_ciegos


def _profundidad_volumetrica_objeto(
    img_rgb: np.ndarray,
    img_lab: np.ndarray,
    enfoque: np.ndarray,
    grad_map: np.ndarray,
    mascara_fg: np.ndarray,
    mapa_clusters: np.ndarray,
    slic_map: np.ndarray | None = None,
    anat_retrato: dict | None = None,
    info_objeto: dict | None = None,
) -> np.ndarray:
    """
    Reconstruye el volumen 3D de cualquier objeto aislado:
      - Si anat_retrato indica que es un rostro/persona, aplica el modelo anatómico 3D.
      - Para vehículos, jarrones, frutas, animales o productos:
          * Combina Transformada de Distancia con bordes cerrados + radio relativo al eje medial
            (esqueleto local) + perfil cilíndrico de revolución en objetos simétricos para dar
            volumen real a cuerpos redondos y cuellos.
          * Sombreado intrínseco invariante al albedo (vidrios oscuros, llantas negras o sombras
            internas NO crean huecos; brillos blancos NO crean picos).
    """
    if anat_retrato is not None and anat_retrato.get("es_retrato", False):
        return _profundidad_anatomica_retrato(img_rgb, img_lab, mascara_fg, anat_retrato)

    # Verificar si se trata de una pieza rígida de caras planas extruidas (ej. pieza mecánica/placa)
    factor_plano, z_plano, mapa_huecos_ciegos = _detectar_objeto_plano_extruido(
        img_lab, grad_map, mascara_fg, mapa_clusters
    )
    if info_objeto is not None:
        info_objeto["factor_plano"] = factor_plano
        info_objeto["mapa_huecos_ciegos"] = mapa_huecos_ciegos
    if factor_plano >= 0.70 and z_plano is not None:
        relieve_suave = _sombreado_intrinseco_invariante_albedo(img_lab, amplitud=0.020)
        z_out = np.where(mascara_fg, np.clip(z_plano + relieve_suave, 0.14, 1.0), 0.0).astype(np.float32)
        return z_out

    H, W = mascara_fg.shape
    dist = _distancia_con_bordes_cerrados(mascara_fg)
    max_d = float(dist.max())
    if max_d < 1e-3:
        return np.zeros_like(enfoque, dtype=np.float32)

    # A. Volumen macro global + volumen relativo al esqueleto local (eje medial)
    r_global = np.clip(dist / max_d, 0.0, 1.0)
    domo_global = np.sqrt(np.clip(2.0 * r_global - r_global * r_global, 0.0, 1.0))
    domo_global = cv2.GaussianBlur(domo_global, (0, 0), sigmaX=max(2.5, max_d * 0.09))

    # Radio local de referencia (permite que partes más angostas como hojas o extremidades tengan curvatura 3D propia)
    sigma_esq = max(7.0, max_d * 0.35)
    dist_ref_local = cv2.GaussianBlur(dist, (0, 0), sigmaX=sigma_esq) * 1.35 + 2.0
    r_local = np.clip(dist / dist_ref_local, 0.0, 1.0)
    domo_local = np.sqrt(np.clip(2.0 * r_local - r_local * r_local, 0.0, 1.0))

    domo_base = 0.68 * domo_global + 0.32 * domo_local
    if domo_base[mascara_fg].max() > 1e-6:
        domo_base /= float(domo_base[mascara_fg].max())

    # A2. Perfil cilíndrico de revolución para objetos simétricos (jarrones, botellas, tazas, etc.)
    ys_fg, xs_fg = np.where(mascara_fg)
    y_min, y_max = int(ys_fg.min()), int(ys_fg.max())
    x_min, x_max = int(xs_fg.min()), int(xs_fg.max())
    alto_obj = max(y_max - y_min, 1)
    ancho_obj = max(x_max - x_min, 1)

    perfil_rev, simetria_lr = _perfil_revolucion_objeto(mascara_fg)
    if simetria_lr > 0.65 and alto_obj >= 0.72 * ancho_obj:
        peso_rev = float(np.clip((simetria_lr - 0.65) / 0.25, 0.0, 1.0)) * 0.58
        domo_base = (1.0 - peso_rev) * domo_base + peso_rev * perfil_rev
        if domo_base[mascara_fg].max() > 1e-6:
            domo_base /= float(domo_base[mascara_fg].max())

    # B. Rampa de perspectiva 3D y perfil estructurado (ej. vehículos/objetos horizontales en vista 3/4)
    hay_col = mascara_fg.any(axis=0)
    fila_inf_col = H - 1 - np.argmax(mascara_fg[::-1, :], axis=0)
    cols_validas = np.where(hay_col)[0]
    es_horizontal_34 = False
    if len(cols_validas) > 5 and ancho_obj > 1.15 * alto_obj:
        y_base_col = np.interp(
            np.arange(W),
            cols_validas,
            fila_inf_col[cols_validas].astype(np.float32),
        )
        y_base_col = cv2.GaussianBlur(y_base_col.reshape(1, W), (1, 41), 0).ravel()
        b_min, b_max = float(y_base_col[cols_validas].min()), float(y_base_col[cols_validas].max())
        pendiente_base = (b_max - b_min) / float(alto_obj)
        if pendiente_base > 0.06:
            persp_x = np.clip((y_base_col - b_min) / (b_max - b_min + 1e-6), 0.0, 1.0)
        else:
            persp_x = np.full(W, 0.5, dtype=np.float32)
        grid_y, _ = np.mgrid[0:H, 0:W]
        rel_y = np.clip((grid_y - y_min) / float(alto_obj), 0.0, 1.0)
        rampa_perspectiva = 0.62 * persp_x[None, :] + 0.38 * (1.0 - 0.35 * (1.0 - rel_y))
        rampa_perspectiva = cv2.GaussianBlur(rampa_perspectiva.astype(np.float32), (21, 21), 0)

        # En objetos horizontales en perspectiva 3/4 (autos, camionetas, muebles), aplanar ligeramente
        # la cúspide del domo para evitar que las puertas/costados se abomben como un globo
        if ancho_obj > 1.25 * alto_obj and simetria_lr < 0.72 and pendiente_base > 0.08:
            es_horizontal_34 = True
            r_caja = np.clip(r_global / 0.48, 0.0, 1.0)
            perfil_caja = np.sqrt(np.clip(2.0 * r_caja - r_caja * r_caja, 0.0, 1.0))
            perfil_caja = cv2.GaussianBlur(perfil_caja.astype(np.float32), (7, 7), 0)
            domo_base = 0.42 * domo_base + 0.58 * (0.65 * perfil_caja + 0.35 * domo_local)
            if domo_base[mascara_fg].max() > 1e-6:
                domo_base /= float(domo_base[mascara_fg].max())
            peso_rampa = 0.28
        else:
            peso_rampa = 0.16
    else:
        rampa_perspectiva = np.full((H, W), 0.5, dtype=np.float32)
        peso_rampa = 0.0

    if info_objeto is not None:
        info_objeto["es_horizontal_34"] = es_horizontal_34

    # C. Micro-relieve intrínseco invariante al albedo (da relieve a semillas, faros, parrillas y pliegues sin hundir colores oscuros)
    relieve_intrinseco = _sombreado_intrinseco_invariante_albedo(img_lab, amplitud=0.065)

    z_combinado = (1.0 - peso_rampa) * domo_base + peso_rampa * rampa_perspectiva + relieve_intrinseco
    z_combinado = cv2.GaussianBlur(z_combinado.astype(np.float32), (5, 5), 0)

    envolvente_borde = np.sqrt(np.clip(r_global * 3.2, 0.0, 1.0))
    z = np.zeros((H, W), dtype=np.float32)
    if mascara_fg.any():
        vals = z_combinado[mascara_fg]
        lo, hi = np.percentile(vals, [1, 99.5])
        z_norm = np.clip((z_combinado - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        z[mascara_fg] = np.clip(0.14 + 0.86 * z_norm[mascara_fg] * envolvente_borde[mascara_fg], 0.14, 1.0)
    return z


# --------------------------------------------------------------------------------------
# 2. Profundidad para Escenas Completas: Paisajes, Casas, Calles, Interiores y Fotos Generales
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
    slic_map: np.ndarray,
    hipervectores: np.ndarray,
    mascara_sujeto: np.ndarray | None = None,
    anat_retrato: dict | None = None,
) -> np.ndarray:
    """
    Motor de profundidad monocular multi-pista para paisajes, arquitectura/casas y escenas completas,
    con invarianza al albedo para que ventanas oscuras, árboles oscuros o sombras no se hundan.
    """
    if anat_retrato is not None and anat_retrato.get("es_retrato", False) and mascara_sujeto is not None:
        z_ret = _profundidad_anatomica_retrato(img_rgb, img_lab, mascara_sujeto, anat_retrato)
        z_bg = 0.08 + 0.20 * y_norm
        z_bg[es_cielo] = 0.0
        return np.where(mascara_sujeto, np.maximum(z_ret, 0.35), z_bg).astype(np.float32)

    H, W = y_norm.shape
    a_c = img_lab[:, :, 1] - 128.0
    b_c = img_lab[:, :, 2] - 128.0

    # --- Pista 1: Perspectiva atmosférica (Haze / Contraste de saturación y calidez, sin confundir objetos oscuros con lejanos) ---
    img_f = img_rgb.astype(np.float32) / 255.0
    canal_oscuro = np.min(img_f, axis=2)
    canal_oscuro = cv2.erode(canal_oscuro, np.ones((7, 7), np.uint8))
    cercania_atmosferica = 1.0 - cv2.GaussianBlur(canal_oscuro, (19, 19), 0)
    calidez = np.clip((b_c - a_c * 0.3 + 20.0) / 60.0, 0.0, 1.0)
    pista_atmosfera = _normalizar(0.68 * cercania_atmosferica + 0.32 * calidez, 2, 98)

    # --- Pista 2: Nitidez / Textura Multi-escala y Saliencia ---
    pista_nitidez = _pista_nitidez_multiescala(img_lab, enfoque, grad_map)
    pista_saliencia = _pista_saliencia_superpixeles(slic_map, hipervectores)

    # --- Pista 3: Geometría del plano de tierra, montañas y perspectiva hacia el punto de fuga ---
    if es_cielo.any():
        filas_cielo = np.where(es_cielo.mean(axis=1) > 0.25)[0]
        if len(filas_cielo) > 0:
            y_h_efectivo = float(np.clip(filas_cielo.max() / float(H), 0.12, 0.65))
        else:
            y_h_efectivo = y_horizonte
    else:
        y_h_efectivo = y_horizonte

    z_suelo = _prior_suelo(H, W, y_h_efectivo - 0.03)
    efecto_vp = np.abs(x_norm - x_vp) * (1.0 - y_norm) * (0.26 if vp_confianza > 0.25 else 0.10)
    z_perspectiva = np.clip(z_suelo * 0.92 + efecto_vp + 0.04, 0.04, 1.0)

    sobre_horiz = (y_norm <= y_h_efectivo) & (~es_cielo)
    dist_sobre_h = np.clip((y_h_efectivo - y_norm) / max(y_h_efectivo, 0.15), 0.0, 1.0)
    z_sobre_h = np.clip(
        0.26 * (1.0 - 0.55 * dist_sobre_h) + 0.44 * pista_atmosfera + 0.30 * pista_nitidez,
        0.06, 0.72
    )
    z_base_escena = np.where(sobre_horiz, z_sobre_h, z_perspectiva)

    # --- Pista 4: Extrusión vertical coherente de casas, edificios, árboles y estructuras ---
    L_norm = img_lab[:, :, 0].astype(np.float32) / 255.0
    gx = cv2.Sobel(L_norm, cv2.CV_32F, 1, 0, ksize=3)
    borde_v = np.abs(gx) / (np.percentile(np.abs(gx), 98) + 1e-6)
    contraste_local = cv2.GaussianBlur(grad_map, (9, 9), 0)

    es_estructura = (~es_cielo) & (
        ((y_norm < y_h_efectivo + 0.04) & ((pista_atmosfera > 0.45) | (pista_saliencia > 0.55))) |
        ((y_norm >= y_h_efectivo - 0.02) & (y_norm < 0.94) & (
            ((enfoque > 0.24) & (borde_v > 0.22)) |
            (contraste_local > 0.24) |
            (pista_saliencia > 0.58) |
            ((a_c < -8.0) & (enfoque > 0.20))
        ))
    )
    es_estructura = cv2.morphologyEx(
        es_estructura.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 35) if tipo_escena == "arquitectura" else (11, 19))
    ) > 0
    if tipo_escena == "arquitectura":
        # En fachadas arquitectónicas, las ventanas oscuras y muros lisos forman parte de un mismo
        # plano vertical continuo desde el techo hasta la base (sin escalones en el horizonte)
        es_estructura = ndi.binary_fill_holes(es_estructura)
        # Propagar hacia abajo cualquier columna con fachada sobre el horizonte hasta la zona de suelo
        tiene_techo = (es_estructura & (y_norm < y_h_efectivo + 0.04)).any(axis=0)
        if tiene_techo.any():
            for x_col in np.where(tiene_techo)[0]:
                ys_c = np.where(es_estructura[:, x_col])[0]
                if len(ys_c) >= 2:
                    es_estructura[ys_c[0]:ys_c[-1] + 1, x_col] = True
    es_estructura &= (~es_cielo)

    base_valida = es_estructura & (y_norm >= y_h_efectivo - 0.02)
    hay_base = base_valida.any(axis=0)
    fila_base = H - 1 - np.argmax(base_valida[::-1, :], axis=0)
    z_raiz_col = np.where(
        hay_base,
        z_perspectiva[fila_base, np.arange(W)],
        z_base_escena.mean(axis=0)
    ).astype(np.float32)
    z_raiz_col = cv2.GaussianBlur(z_raiz_col.reshape(1, W), (1, 25), 0).ravel()
    z_estruct = np.where(es_estructura, z_raiz_col[None, :], 0.0).astype(np.float32)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    z_prop = z_estruct.copy()
    for _ in range(12):
        z_dil = cv2.dilate(z_prop, kern)
        z_prop = np.where(es_estructura, np.maximum(z_prop, z_dil * 0.995), 0.0)

    relieve_intrinseco = _sombreado_intrinseco_invariante_albedo(img_lab, amplitud=0.045)

    z = (
        0.52 * z_base_escena +
        0.20 * pista_atmosfera +
        0.15 * pista_nitidez +
        0.13 * pista_saliencia +
        relieve_intrinseco
    )
    if es_estructura.any():
        peso_est = 0.88 if tipo_escena == "arquitectura" else 0.65
        z_est_completo = np.clip(z_prop + relieve_intrinseco, 0.08, 1.0)
        if tipo_escena == "arquitectura":
            # Mantener la fachada arquitectónica como un plano vertical rígido y coherente
            z[es_estructura] = (
                (1.0 - peso_est) * z[es_estructura] +
                peso_est * z_est_completo[es_estructura]
            )
        else:
            z[es_estructura] = (
                (1.0 - peso_est) * z[es_estructura] +
                peso_est * np.maximum(z_est_completo[es_estructura], z[es_estructura])
            )

    if mascara_sujeto is not None and tipo_escena != "arquitectura":
        frac_s = float(mascara_sujeto.mean())
        if 0.03 <= frac_s <= 0.78:
            dist_s = _distancia_con_bordes_cerrados(mascara_sujeto)
            domo_s = np.sqrt(np.clip(dist_s / (float(dist_s.max()) + 1e-6), 0.0, 1.0))
            domo_s = cv2.GaussianBlur(domo_s, (11, 11), 0)
            ys_s = np.where(mascara_sujeto)[0]
            y_apoyo = float(np.percentile(ys_s, 95)) / float(H)
            z_apoyo = float(np.clip((y_apoyo - y_h_efectivo) / (1.0 - y_h_efectivo + 1e-6), 0.38, 0.88))
            z_sujeto = np.clip(z_apoyo + 0.24 * domo_s + relieve_intrinseco, 0.25, 1.0)
            z = np.where(mascara_sujeto, 0.35 * z + 0.65 * np.maximum(z, z_sujeto), z)

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
    anat_retrato = info_escena.get("anatomia_retrato", None)

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
    mapa_cavidad = None
    mapa_realce = None
    info_objeto: dict = {"factor_plano": 0.0, "mapa_huecos_ciegos": None}
    modo = f"no supervisado ({tipo_escena})"
    if mascara_fg is not None:
        es_cielo = ~mascara_fg
        mapa_geom = _profundidad_volumetrica_objeto(
            img_rgb, img_lab, enfoque, grad_map, mascara_fg, mapa_clusters, slic_map, anat_retrato,
            info_objeto=info_objeto,
        )
        if info_objeto.get("mapa_huecos_ciegos") is not None:
            mapa_cavidad = info_objeto["mapa_huecos_ciegos"]
        elif tipo_escena != "retrato" and info_objeto.get("factor_plano", 0.0) < 0.5:
            cav = _detectar_cavidad_superior(img_lab, grad_map, mascara_fg)
            if float(cav.max()) > 0.15:
                mapa_cavidad = cav
            elif not info_objeto.get("es_horizontal_34", False):
                realce_boq, cav_boq = _detectar_boquilla_frontal(img_lab, grad_map, mascara_fg)
                if realce_boq is not None and cav_boq is not None:
                    # Escalar levemente el cuerpo base para que el collarín de la boquilla sobresalga con claridad
                    mapa_geom = np.where(
                        mascara_fg,
                        np.clip(0.76 * mapa_geom + 0.34 * realce_boq, 0.14, 1.0),
                        0.0,
                    ).astype(np.float32)
                    mapa_realce = realce_boq
                    mapa_cavidad = cav_boq
    else:
        es_cielo = _mascara_cielo(img_lab, enfoque, y_norm, y_h)
        usa_suelo = True
        mapa_geom = _profundidad_escena_universal(
            img_rgb, img_lab, enfoque, grad_map, y_norm, x_norm,
            y_h, x_vp, es_cielo, vp_conf, tipo_escena,
            slic_map=slic_map, hipervectores=hipervectores,
            mascara_sujeto=mascara_sujeto_escena,
            anat_retrato=anat_retrato,
        )

    factor_plano = float(info_objeto.get("factor_plano", 0.0))
    info_escena["mapa_cavidad"] = mapa_cavidad
    info_escena["mapa_realce"] = mapa_realce
    info_escena["factor_plano"] = factor_plano
    info_escena["es_horizontal_34"] = bool(info_objeto.get("es_horizontal_34", False))

    # 3. Regularización por superpíxeles (Make3D) sin destruir la continuidad anatómica
    planos = _ajuste_planos_por_superpixel(mapa_geom, slic_map)
    if tipo_escena == "retrato":
        peso_planos = 0.12
    elif tipo_escena == "arquitectura" or factor_plano >= 0.70:
        peso_planos = 0.45
    else:
        peso_planos = 0.28
    mapa_z_bruto = ((1.0 - peso_planos) * mapa_geom + peso_planos * planos).astype(np.float32)

    # 4. Refinamiento final de bordes SIN contaminar la profundidad con el albedo (colores oscuros/claros)
    if mascara_fg is not None:
        if factor_plano >= 0.70:
            # Pieza rígida/plana: preservar aristas nítidas y meseta sin abombamiento
            z_suave = cv2.GaussianBlur(mapa_z_bruto, (3, 3), 0)
            mapa_z_final = np.where(mascara_fg, np.clip(z_suave, 0.15, 1.0), 0.0).astype(np.float32)
        else:
            # En objetos curvos y retratos aislados, suavizamos geométricamente dentro de la máscara
            z_suave = cv2.GaussianBlur(mapa_z_bruto, (7, 7), 0)
            z_guiado = filtro_guiado_rapido(img_rgb, mapa_z_bruto, radio=3, eps=2e-2)
            mapa_z_final = (0.82 * z_suave + 0.18 * z_guiado).astype(np.float32)
            if mascara_fg.any():
                vals = mapa_z_final[mascara_fg]
                lo, hi = np.percentile(vals, [1, 99.5])
                z_fg = np.clip((mapa_z_final - lo) / (hi - lo + 1e-6), 0.0, 1.0)
                dist_borde = _distancia_con_bordes_cerrados(mascara_fg)
                peso_borde = np.clip(dist_borde / max(float(dist_borde.max()) * 0.18, 2.5), 0.0, 1.0)
                mapa_z_final = np.where(mascara_fg, 0.15 + 0.85 * z_fg * np.sqrt(peso_borde), 0.0).astype(np.float32)
                if mapa_cavidad is not None:
                    mapa_z_final = np.where(
                        mascara_fg,
                        np.clip(mapa_z_final - 0.35 * mapa_cavidad, 0.05, 1.0),
                        0.0,
                    ).astype(np.float32)
            else:
                mapa_z_final = _normalizar(mapa_z_final)
    else:
        # En escenas completas, mezclamos suavizado bilateral con filtro guiado suave de alta regularización (eps=8e-3)
        z_guiado = filtro_guiado_rapido(img_rgb, mapa_z_bruto, radio=5, eps=8e-3)
        z_bilat = cv2.bilateralFilter(mapa_z_bruto, d=7, sigmaColor=0.10, sigmaSpace=7)
        mapa_z_final = _normalizar(0.55 * z_bilat + 0.45 * z_guiado)
        mapa_z_final[es_cielo] = np.minimum(mapa_z_final[es_cielo], 0.02)

    metricas = {
        "modo_profundidad": modo,
        "tipo_escena": tipo_escena,
        "suelo_en_perspectiva": usa_suelo,
        "fondo_eliminado": mascara_fg is not None,
        "objeto_plano": factor_plano >= 0.70,
        "cavidad_detectada": mapa_cavidad is not None,
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
