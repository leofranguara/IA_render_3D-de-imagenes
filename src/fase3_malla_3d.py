"""
Fase 3: Construcción de malla 3D volumétrica universal con soporte para:
  - Objetos aislados (autos, frutas, figuras, muebles) con volumen 3D cerrado en espejo.
  - Escenas completas (paisajes, casas, calles) con relieve topográfico suave, proyección
    libre de distorsión trapezoidal extrema y cierre trasero simétrico en espejo.
"""
import os

import cv2
import numpy as np
import trimesh
from scipy import ndimage as ndi


def disparidad_a_profundidad(d: np.ndarray, z_cerca: float = 1.0, z_lejos: float = 8.0) -> np.ndarray:
    inv = 1.0 / z_lejos + np.clip(d, 0.0, 1.0) * (1.0 / z_cerca - 1.0 / z_lejos)
    return (1.0 / inv).astype(np.float32)


def construir_malla_volumetrica(
    img_rgb: np.ndarray,
    mapa_z: np.ndarray,
    mapa_clusters: np.ndarray | None = None,
    y_horizonte: float = 0.45,
    paso_malla: int = 1,
    fov_grados: float = 60.0,
    z_cerca: float = 1.0,
    z_lejos: float = 8.0,
    umbral_discontinuidad: float = 0.20,
    area_minima_px: int = 150,
    mascara_fg: np.ndarray | None = None,
    relleno_espejo: bool = True,
    grosor_espejo: float = 0.65,
    mapa_cavidad: np.ndarray | None = None,
    mapa_realce: np.ndarray | None = None,
    factor_plano: float = 0.0,
    es_horizontal_34: bool = False,
) -> trimesh.Trimesh:
    """
    Construye una malla 3D a partir del mapa de disparidad/profundidad:
      - Si mascara_fg está activa (objeto sin fondo): genera un cuerpo 3D proporcionado y suave.
      - Si mascara_fg es None (escena completa: paisaje, casa con entorno, calle): usa proyección
        híbrida equilibrada para que el fondo no se deforme ni se rompa al rotar o al aplicar el
        relleno en espejo trasero.
      - Si relleno_espejo=True: duplica la geometría hacia atrás como un espejo simétrico y cose
        automáticamente todos los bordes del perímetro para obtener un sólido cerrado.
      - grosor_espejo: factor multiplicador del espesor en modo espejo (ej. 0.15 = muy delgado,
        0.65 = equilibrado, 1.0 = normal, 1.5 = muy grueso).
    """
    img = img_rgb[::paso_malla, ::paso_malla]
    d = mapa_z[::paso_malla, ::paso_malla].astype(np.float32)
    fg = mascara_fg[::paso_malla, ::paso_malla].astype(bool) if mascara_fg is not None else None
    H, W = d.shape
    factor_grosor = float(np.clip(grosor_espejo, 0.08, 2.0))

    f = 0.5 * W / np.tan(np.deg2rad(fov_grados) / 2.0)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    if fg is not None and fg.any():
        # --- MODO 1: Objeto Aislado (Retrato, Auto, Fruta, Casa recortada, Persona) ---
        z_pivote = 2.2
        escala_xy = z_pivote / f
        X = (u - cx) * escala_xy
        Y = -(v - cy) * escala_xy

        # Transformada de distancia con bordes cerrados en los 4 lados del marco
        m_pad = np.zeros((H + 2, W + 2), dtype=np.uint8)
        m_pad[1:-1, 1:-1] = fg.astype(np.uint8)
        dist_cerrada = cv2.distanceTransform(m_pad, cv2.DIST_L2, 5)[1:-1, 1:-1].astype(np.float32)
        dist_abierta = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        dist_px = 0.80 * dist_cerrada + 0.20 * dist_abierta
        max_dist = max(float(dist_cerrada.max()), 1.0)

        # El espesor físico 3D debe ser proporcional al diámetro transversal inscrito del sujeto
        # (evita que en retratos verticales o personas de cuerpo entero la profundidad se estire por la altura del torso)
        ys_fg, xs_fg = np.where(fg)
        ancho_fg = float(xs_fg.max() - xs_fg.min())
        alto_fg = float(ys_fg.max() - ys_fg.min())
        diametro_transversal = min(max(2.3 * max_dist, 32.0), max(min(ancho_fg, alto_fg) * 1.15, 32.0))
        es_plano = factor_plano >= 0.70
        if es_plano:
            coef_espesor = 0.24
        elif es_horizontal_34:
            coef_espesor = 0.32
        else:
            coef_espesor = 0.48
        semiespesor_max = coef_espesor * diametro_transversal * escala_xy * factor_grosor

        # Bisel corto para piezas rígidas/planas (aristas nítidas) o bisel suave para objetos curvos
        if es_plano:
            frac_bisel = 0.05
        elif es_horizontal_34:
            frac_bisel = 0.12
        else:
            frac_bisel = 0.18
        r_borde = np.clip(dist_px / (max_dist * frac_bisel + 1e-6), 0.0, 1.0)
        envolvente = np.sqrt(np.clip(2.0 * r_borde - r_borde * r_borde, 0.0, 1.0))

        k_blur = (3, 3) if es_plano else (5, 5)
        d_suave = cv2.GaussianBlur(d, k_blur, 0)
        d_norm = np.clip((d_suave - 0.12) / 0.88, 0.0, 1.0)
        dz = semiespesor_max * (0.10 * envolvente + 0.90 * d_norm * envolvente)
        dz = cv2.GaussianBlur(dz.astype(np.float32), k_blur, 0)
        dz[~fg] = 0.0

        # Extrusión frontal de boquillas/tubos salientes (sobresale solo por delante, sin deformar la espalda)
        dz_frente = dz.copy()
        if mapa_realce is not None:
            realce = mapa_realce[::paso_malla, ::paso_malla].astype(np.float32)
            realce = cv2.GaussianBlur(realce, (5, 5), 0)
            borde_suave = np.sqrt(np.clip(envolvente * 2.5, 0.0, 1.0))
            dz_frente = np.where(
                fg,
                dz_frente + semiespesor_max * 0.75 * realce * borde_suave,
                0.0,
            ).astype(np.float32)

        # Si existe una cavidad interior (ej. boca de un jarrón o conducto de boquilla), hundir solo la cara frontal
        # manteniendo la pared trasera convexa para conservar el volumen exterior sin atravesar la malla
        if mapa_cavidad is not None:
            cav = mapa_cavidad[::paso_malla, ::paso_malla].astype(np.float32)
            cav = cv2.GaussianBlur(cav, (5, 5), 0)
            prof_hueco = semiespesor_max * 1.15 * cav
            dz_frente = np.where(fg, np.maximum(dz_frente - prof_hueco, -dz * 0.72), 0.0).astype(np.float32)

        z_frente = -z_pivote + dz_frente
        z_detras = -z_pivote - dz
        valido = fg[:-1, :-1] & fg[:-1, 1:] & fg[1:, :-1] & fg[1:, 1:]
    else:
        # --- MODO 2: Escena Completa (Paisajes, Casas con entorno, Calles, Interiores) ---
        z_pivote = 2.4
        escala_base = z_pivote / f

        d_suave = cv2.bilateralFilter(d, d=7, sigmaColor=0.07, sigmaSpace=5)
        d_suave = cv2.GaussianBlur(d_suave, (3, 3), 0)

        if relleno_espejo:
            # Con relleno en espejo, creamos un diorama 3D volumétrico de doble cara con grosor ajustable:
            factor_persp = 1.0 - 0.18 * (d_suave - 0.5)
            X = (u - cx) * escala_base * factor_persp
            Y = -(v - cy) * escala_base * factor_persp

            span_escena = max(W, H) * escala_base
            espesor_base = 0.020 * span_escena * factor_grosor
            relieve_max = 0.26 * span_escena * factor_grosor

            # Atenuar ligeramente en el borde mismo del marco para un bisel limpio
            margen_x = np.minimum(u, W - 1 - u) / max(W * 0.04, 1.0)
            margen_y = np.minimum(v, H - 1 - v) / max(H * 0.04, 1.0)
            bisel_marco = np.clip(np.minimum(margen_x, margen_y), 0.15, 1.0)

            dz = (espesor_base + relieve_max * (d_suave ** 1.10)) * bisel_marco
            z_frente = -z_pivote + dz
            z_detras = -z_pivote - dz

            # En modo espejo de escena completa, mantenemos la malla continua sin agujeros internos
            # salvo que el usuario baje explícitamente el umbral por debajo de 0.12
            if umbral_discontinuidad < 0.12:
                d00, d01, d10, d11 = d_suave[:-1, :-1], d_suave[:-1, 1:], d_suave[1:, :-1], d_suave[1:, 1:]
                dmax = np.maximum(np.maximum(d00, d01), np.maximum(d10, d11))
                dmin = np.minimum(np.minimum(d00, d01), np.minimum(d10, d11))
                valido = (dmax - dmin) < (umbral_discontinuidad * 2.0)
            else:
                valido = np.ones((H - 1, W - 1), dtype=bool)
        else:
            # Superficie 2.5D con cámara pinhole suavizada (z_lejos moderado para no deformar el cielo)
            z_lejos_ef = min(z_lejos, 3.8)
            Z = disparidad_a_profundidad(d_suave, z_cerca, z_lejos_ef)
            z_pivote = float(np.percentile(Z, 45))
            X = (u - cx) / f * Z
            Y = -(v - cy) / f * Z
            z_frente = -Z
            z_detras = -Z

            z00, z01, z10, z11 = Z[:-1, :-1], Z[:-1, 1:], Z[1:, :-1], Z[1:, 1:]
            zmax = np.maximum(np.maximum(z00, z01), np.maximum(z10, z11))
            zmin = np.minimum(np.minimum(z00, z01), np.minimum(z10, z11))
            umbral = max(umbral_discontinuidad, 0.25) * paso_malla
            valido = (zmax - zmin) / zmin < umbral

    # Quita islas diminutas (ruido flotante)
    etiquetas, n = ndi.label(valido)
    if n > 0:
        tam = np.bincount(etiquetas.ravel())
        valido &= (tam[etiquetas] >= min(area_minima_px, max(20, int(valido.sum() * 0.01))))

    N = H * W
    idx = np.arange(N).reshape(H, W)
    a = idx[:-1, :-1][valido]
    b = idx[:-1, 1:][valido]
    c = idx[1:, :-1][valido]
    e = idx[1:, 1:][valido]
    caras_frente = np.vstack([np.column_stack([a, c, e]), np.column_stack([a, e, b])])

    verts_frente = np.column_stack([X.ravel(), Y.ravel(), z_frente.ravel()])
    colores_base = np.column_stack([img.reshape(-1, 3), np.full(N, 255, np.uint8)]).astype(np.uint8)

    if relleno_espejo:
        verts_detras = np.column_stack([X.ravel(), Y.ravel(), z_detras.ravel()])
        # Invertir el orden de vértices en la cara trasera para que las normales apunten hacia afuera
        caras_detras = caras_frente[:, [0, 2, 1]] + N

        # Coser automáticamente todas las aristas frontera entre la cara frontal y su espejo trasero
        v_pad = np.pad(valido, 1, mode="constant", constant_values=False)
        caras_costura = []

        m_top = valido & (~v_pad[:-2, 1:-1])
        if m_top.any():
            u0, u1 = idx[:-1, :-1][m_top], idx[:-1, 1:][m_top]
            caras_costura.append(np.column_stack([u0, u1, u1 + N]))
            caras_costura.append(np.column_stack([u0, u1 + N, u0 + N]))

        m_bot = valido & (~v_pad[2:, 1:-1])
        if m_bot.any():
            d0, d1 = idx[1:, 1:][m_bot], idx[1:, :-1][m_bot]
            caras_costura.append(np.column_stack([d0, d1, d1 + N]))
            caras_costura.append(np.column_stack([d0, d1 + N, d0 + N]))

        m_left = valido & (~v_pad[1:-1, :-2])
        if m_left.any():
            l0, l1 = idx[1:, :-1][m_left], idx[:-1, :-1][m_left]
            caras_costura.append(np.column_stack([l0, l1, l1 + N]))
            caras_costura.append(np.column_stack([l0, l1 + N, l0 + N]))

        m_right = valido & (~v_pad[1:-1, 2:])
        if m_right.any():
            r0, r1 = idx[:-1, 1:][m_right], idx[1:, 1:][m_right]
            caras_costura.append(np.column_stack([r0, r1, r1 + N]))
            caras_costura.append(np.column_stack([r0, r1 + N, r0 + N]))

        verts = np.vstack([verts_frente, verts_detras])
        if caras_costura:
            caras = np.vstack([caras_frente, caras_detras, np.vstack(caras_costura)])
        else:
            caras = np.vstack([caras_frente, caras_detras])
        colores = np.vstack([colores_base, colores_base])
    else:
        verts = verts_frente
        caras = caras_frente
        colores = colores_base

    malla = trimesh.Trimesh(vertices=verts, faces=caras, vertex_colors=colores, process=False)
    malla.remove_unreferenced_vertices()
    malla.metadata.update({
        "f": float(f), "W": int(W), "H": int(H),
        "z_pivote": float(z_pivote),
        "relleno_espejo": bool(relleno_espejo),
        "grosor_espejo": float(factor_grosor),
    })
    return malla


# --------------------------------------------------------------------------------------
# Vistas desde otros ángulos (sin necesidad de ventana ni GPU) para verificar la geometría
# --------------------------------------------------------------------------------------
def renderizar_vista(malla: trimesh.Trimesh, yaw_grados: float = 0.0, pitch_grados: float = 0.0,
                     tam_punto: int = 2) -> np.ndarray:
    """
    Renderiza la malla de triángulos desde una cámara rotada alrededor del centro de la escena
    usando ordenamiento por profundidad de caras (Painter's Algorithm sobre triángulos rellenos),
    para que al girar el modelo no se transparente la cara trasera a través de la delantera.
    """
    f, W, H = malla.metadata["f"], malla.metadata["W"], malla.metadata["H"]
    piv = np.array([0.0, 0.0, -malla.metadata["z_pivote"]])
    V = np.asarray(malla.vertices) - piv
    F = np.asarray(malla.faces)
    col_v = np.asarray(malla.visual.vertex_colors)[:, :3]

    ya, pa = np.deg2rad(yaw_grados), np.deg2rad(pitch_grados)
    Ry = np.array([[np.cos(ya), 0, np.sin(ya)], [0, 1, 0], [-np.sin(ya), 0, np.cos(ya)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pa), -np.sin(pa)], [0, np.sin(pa), np.cos(pa)]])
    P = (Rx @ Ry @ V.T).T + piv

    # Para velocidad en vistas.png, submuestreamos o usamos rasterización densa con Z-buffer por píxel
    prof = -P[:, 2]
    ok_v = prof > 0.05
    u = np.round(f * P[:, 0] / np.maximum(prof, 0.05) + (W - 1) / 2).astype(np.int32)
    v = np.round(-f * P[:, 1] / np.maximum(prof, 0.05) + (H - 1) / 2).astype(np.int32)

    # Descartar caras cuyas normales apuntan en la misma dirección que el rayo de visión (Back-face culling)
    v0, v1, v2 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    normales = np.cross(v1 - v0, v2 - v0)
    centro_cara = (v0 + v1 + v2) / 3.0
    # El vector de la cámara (en el origen 0,0,0) hacia la cara es centro_cara; visible si dot(normal, centro_cara) < 0
    visibles = (np.sum(normales * centro_cara, axis=1) < 0) & ok_v[F[:, 0]] & ok_v[F[:, 1]] & ok_v[F[:, 2]]

    F_vis = F[visibles]
    if len(F_vis) == 0:
        F_vis = F

    # Muestrear puntos en los vértices y baricentros de las caras visibles ordenados de lejos a cerca
    c_u = ((u[F_vis[:, 0]] + u[F_vis[:, 1]] + u[F_vis[:, 2]]) // 3)
    c_v = ((v[F_vis[:, 0]] + v[F_vis[:, 1]] + v[F_vis[:, 2]]) // 3)
    c_prof = (prof[F_vis[:, 0]] + prof[F_vis[:, 1]] + prof[F_vis[:, 2]]) / 3.0
    c_col = ((col_v[F_vis[:, 0]].astype(np.int32) + col_v[F_vis[:, 1]] + col_v[F_vis[:, 2]]) // 3).astype(np.uint8)

    verts_vis = np.unique(F_vis.ravel())
    all_u = np.concatenate([u[verts_vis], c_u])
    all_v = np.concatenate([v[verts_vis], c_v])
    all_prof = np.concatenate([prof[verts_vis], c_prof])
    all_col = np.vstack([col_v[verts_vis], c_col])

    lienzo = np.full((H, W, 3), 28, np.uint8)
    orden = np.argsort(-all_prof)
    all_u, all_v, all_col = all_u[orden], all_v[orden], all_col[orden]
    for du in range(tam_punto):
        for dv in range(tam_punto):
            uu, vv = all_u + du, all_v + dv
            m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            lienzo[vv[m], uu[m]] = all_col[m]
    return lienzo


def guardar_vistas(malla: trimesh.Trimesh, ruta_png: str, yaws=(-45, -20, 0, 20, 45, 135, 180), pitch=0.0):
    tiras = [renderizar_vista(malla, y, pitch) for y in yaws]
    cv2.imwrite(ruta_png, cv2.cvtColor(np.hstack(tiras), cv2.COLOR_RGB2BGR))


def visualizar_y_exportar(malla: trimesh.Trimesh, ruta_obj: str = "data/modelo_3d.obj",
                          mostrar: bool = True):
    carpeta = os.path.dirname(ruta_obj) or "."
    os.makedirs(carpeta, exist_ok=True)
    base = os.path.splitext(ruta_obj)[0]
    malla.export(ruta_obj)
    malla.export(base + ".ply")
    malla.export(base + ".glb")
    print(f"\n[OK] Escenario 3D exportado en: {ruta_obj} (+ .ply, .glb)")
    print(f"• Vértices: {len(malla.vertices):,} | Caras: {len(malla.faces):,}")

    if not mostrar:
        return
    try:
        import open3d as o3d
    except ImportError:
        print("[i] open3d no está instalado: abre el .ply/.glb en MeshLab o Blender.")
        return

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(malla.vertices)
    m.triangles = o3d.utility.Vector3iVector(malla.faces)
    m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(malla.visual.vertex_colors)[:, :3] / 255.0)
    m.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Reconstrucción 3D desde una imagen", width=1200, height=820)
    vis.add_geometry(m)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.11, 0.11, 0.13])
    opt.mesh_show_back_face = True
    opt.light_on = False
    ctr = vis.get_view_control()
    zp = malla.metadata.get("z_pivote", 2.0)
    ctr.set_lookat([0.0, 0.0, -zp])
    ctr.set_front([0.0, 0.0, 1.0])
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_zoom(0.5)
    vis.run()
    vis.destroy_window()
