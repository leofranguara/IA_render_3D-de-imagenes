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
def _profundidad_volumetrica_objeto(
    img_rgb: np.ndarray,
    img_lab: np.ndarray,
    enfoque: np.ndarray,
    grad_map: np.ndarray,
    mascara_fg: np.ndarray,
    mapa_clusters: np.ndarray,
    slic_map: np.ndarray | None = None,
    anat_retrato: dict | None = None,
) -> np.ndarray:
    """
    Reconstruye el volumen 3D de cualquier objeto aislado:
      - Si anat_retrato indica que es un rostro/persona, aplica el modelo anatómico 3D.
      - Para vehículos, frutas, animales o productos:
          * Combina Transformada de Distancia con bordes cerrados + radio relativo al eje medial
            (esqueleto local) para que tanto el cuerpo principal como partes estrechas (hojas, tallos,
            alerones, extremidades) tengan volumen 3D proporcionado.
          * Sombreado intrínseco invariante al albedo (vidrios oscuros, llantas negras o sombras
            internas NO crean huecos; brillos blancos NO crean picos).
    """
    if anat_retrato is not None and anat_retrato.get("es_retrato", False):
        return _profundidad_anatomica_retrato(img_rgb, img_lab, mascara_fg, anat_retrato)

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

    # B. Rampa suave de perspectiva 3D (solo significativa en objetos horizontales apoyados como automóviles)
    ys_fg, xs_fg = np.where(mascara_fg)
    y_min, y_max = int(ys_fg.min()), int(ys_fg.max())
    x_min, x_max = int(xs_fg.min()), int(xs_fg.max())
    alto_obj = max(y_max - y_min, 1)
    ancho_obj = max(x_max - x_min, 1)

    hay_col = mascara_fg.any(axis=0)
    fila_inf_col = H - 1 - np.argmax(mascara_fg[::-1, :], axis=0)
    cols_validas = np.where(hay_col)[0]
    if len(cols_validas) > 5 and ancho_obj > 1.10 * alto_obj:
        y_base_col = np.interp(
            np.arange(W),
            cols_validas,
            fila_inf_col[cols_validas].astype(np.float32),
        )
        y_base_col = cv2.GaussianBlur(y_base_col.reshape(1, W), (1, 41), 0).ravel()
        b_min, b_max = float(y_base_col[cols_validas].min()), float(y_base_col[cols_validas].max())
        if b_max - b_min > 0.06 * alto_obj:
            persp_x = np.clip((y_base_col - b_min) / (b_max - b_min + 1e-6), 0.0, 1.0)
        else:
            persp_x = np.full(W, 0.5, dtype=np.float32)
        grid_y, _ = np.mgrid[0:H, 0:W]
        rel_y = np.clip((grid_y - y_min) / float(alto_obj), 0.0, 1.0)
        rampa_perspectiva = 0.55 * persp_x[None, :] + 0.45 * (1.0 - 0.30 * (1.0 - rel_y))
        rampa_perspectiva = cv2.GaussianBlur(rampa_perspectiva.astype(np.float32), (21, 21), 0)
        peso_rampa = 0.16
    else:
        rampa_perspectiva = np.full((H, W), 0.5, dtype=np.float32)
        peso_rampa = 0.0

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
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 19))
    ) > 0
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
        peso_est = 0.80 if tipo_escena == "arquitectura" else 0.65
        z_est_completo = np.clip(z_prop + relieve_intrinseco, 0.08, 1.0)
        z[es_estructura] = (
            (1.0 - peso_est) * z[es_estructura] +
            peso_est * np.maximum(z_est_completo[es_estructura], z[es_estructura])
        )

    if mascara_sujeto is not None:
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
                img_rgb, img_lab, enfoque, grad_map, mascara_fg, mapa_clusters, slic_map, anat_retrato
            )
            d = np.where(mascara_fg, 0.50 * d + 0.50 * z_domo, 0.0)
        mapa_geom = d
    else:
        modo = f"universal ({tipo_escena})"
        if mascara_fg is not None:
            es_cielo = ~mascara_fg
            mapa_geom = _profundidad_volumetrica_objeto(
                img_rgb, img_lab, enfoque, grad_map, mascara_fg, mapa_clusters, slic_map, anat_retrato
            )
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

    # 3. Regularización por superpíxeles (Make3D) sin destruir la continuidad anatómica
    planos = _ajuste_planos_por_superpixel(mapa_geom, slic_map)
    if tipo_escena == "retrato":
        peso_planos = 0.12
    elif tipo_escena == "arquitectura":
        peso_planos = 0.45
    else:
        peso_planos = 0.28
    mapa_z_bruto = ((1.0 - peso_planos) * mapa_geom + peso_planos * planos).astype(np.float32)

    # 4. Refinamiento final de bordes SIN contaminar la profundidad con el albedo (colores oscuros/claros)
    if mascara_fg is not None:
        # En objetos y retratos aislados, suavizamos geométricamente dentro de la máscara
        # (NO aplicamos filtro guiado por luminancia intensa para evitar que barbas o vidrios oscuros se hundan)
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
