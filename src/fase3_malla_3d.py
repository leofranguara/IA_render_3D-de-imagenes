"""
Fase 3: del mapa de profundidad a una malla 3D con cámara pinhole.

Cambios clave respecto a la versión anterior (causa de las fallas de perspectiva):
  - Antes: X,Y se escalaban con un "factor de perspectiva" inventado y Z era lineal en la
    disparidad. Eso deforma la escena. Ahora cada píxel se retroproyecta con el modelo de
    cámara real:   X = (u-cx)·Z/f ,  Y = -(v-cy)·Z/f ,  Z = profundidad.
  - La disparidad [0,1] se convierte a profundidad con  1/Z = 1/z_lejos + d·(1/z_cerca - 1/z_lejos)
    (la profundidad es el INVERSO de la disparidad, no una función lineal de ella).
  - Se elimina la "lámina trasera" y las paredes perimetrales: duplicaban la geometría y
    creaban falsos volúmenes. Es una superficie 2.5D honesta.
  - Se cortan los triángulos que unen objetos con el fondo (las "telas de araña" estiradas).
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
    mapa_clusters: np.ndarray | None = None,   # se conserva por compatibilidad
    y_horizonte: float = 0.45,                  # se conserva por compatibilidad
    paso_malla: int = 1,
    fov_grados: float = 60.0,
    z_cerca: float = 1.0,
    z_lejos: float = 8.0,
    umbral_discontinuidad: float = 0.20,
    area_minima_px: int = 150,
    mascara_fg: np.ndarray | None = None,
    relleno_espejo: bool = True,
) -> trimesh.Trimesh:
    """
    Construye una malla 3D a partir del mapa de disparidad/profundidad.
    - Si mascara_fg está activa (fondo eliminado), recorta únicamente el objeto y suaviza el perfil.
    - Si relleno_espejo=True, duplica la geometría hacia atrás como un espejo simétrico y cose
      todo el contorno perimetral para obtener un volumen 3D cerrado por ambos lados.
    """
    img = img_rgb[::paso_malla, ::paso_malla]
    d = mapa_z[::paso_malla, ::paso_malla].astype(np.float32)
    fg = mascara_fg[::paso_malla, ::paso_malla].astype(bool) if mascara_fg is not None else None
    H, W = d.shape

    f = 0.5 * W / np.tan(np.deg2rad(fov_grados) / 2.0)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    if fg is not None and fg.any():
        # Modo objeto aislado (o con relleno espejo): proyección limpia sin deformación cónica en bordes
        z_pivote = 2.2
        escala_xy = z_pivote / f
        X = (u - cx) * escala_xy
        Y = -(v - cy) * escala_xy

        # Espesor proporcional al tamaño físico del objeto en escena
        ys_fg, xs_fg = np.where(fg)
        span_px = max(float(xs_fg.max() - xs_fg.min()), float(ys_fg.max() - ys_fg.min()), 32.0)
        semiespesor_max = 0.36 * span_px * escala_xy

        # Perfil de relieve suave que cae a 0 en el contorno del objeto para un cierre espejo perfecto
        dist_px = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        max_dist = max(float(dist_px.max()), 1.0)
        r_borde = np.clip(dist_px / (max_dist * 0.42 + 1e-6), 0.0, 1.0)
        envolvente = np.sqrt(np.clip(2.0 * r_borde - r_borde * r_borde, 0.0, 1.0))

        d_suave = cv2.bilateralFilter(d, d=7, sigmaColor=0.08, sigmaSpace=5)
        d_norm = np.clip((d_suave - 0.15) / 0.85, 0.0, 1.0)
        dz = semiespesor_max * (0.45 * envolvente + 0.55 * d_norm * envolvente)
        dz = cv2.GaussianBlur(dz.astype(np.float32), (5, 5), 0)
        dz[~fg] = 0.0

        z_frente = -z_pivote + dz
        z_detras = -z_pivote - dz

        # Un cuadro es válido si sus 4 esquinas pertenecen al objeto
        valido = fg[:-1, :-1] & fg[:-1, 1:] & fg[1:, :-1] & fg[1:, 1:]
    else:
        # Escena completa con fondo
        Z = disparidad_a_profundidad(d, z_cerca, z_lejos)
        Z = cv2.bilateralFilter(Z, d=5, sigmaColor=0.05 * float(Z.max()), sigmaSpace=3)
        z_pivote = float(np.percentile(Z, 45))

        X = (u - cx) / f * Z
        Y = -(v - cy) / f * Z
        z_frente = -Z

        # Para el espejo en escena completa, reflejamos respecto al plano base más lejano de cada zona
        z_plano_ref = float(np.percentile(Z, 92))
        dz = np.maximum(z_plano_ref - Z, 0.02)
        z_detras = -z_plano_ref - dz

        z00, z01, z10, z11 = Z[:-1, :-1], Z[:-1, 1:], Z[1:, :-1], Z[1:, 1:]
        zmax = np.maximum(np.maximum(z00, z01), np.maximum(z10, z11))
        zmin = np.minimum(np.minimum(z00, z01), np.minimum(z10, z11))
        umbral = umbral_discontinuidad * paso_malla
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
        # Invertir el orden de los vértices (winding order) en la cara trasera para que las normales apunten hacia afuera
        caras_detras = caras_frente[:, [0, 2, 1]] + N

        # Coser automáticamente todas las aristas frontera entre la malla frontal y su espejo trasero
        v_pad = np.pad(valido, 1, mode="constant", constant_values=False)
        caras_costura = []

        # Aristas superiores (celda válida cuyo vecino de arriba no es válido)
        m_top = valido & (~v_pad[:-2, 1:-1])
        if m_top.any():
            u0, u1 = idx[:-1, :-1][m_top], idx[:-1, 1:][m_top]
            caras_costura.append(np.column_stack([u0, u1, u1 + N]))
            caras_costura.append(np.column_stack([u0, u1 + N, u0 + N]))

        # Aristas inferiores (celda válida cuyo vecino de abajo no es válido)
        m_bot = valido & (~v_pad[2:, 1:-1])
        if m_bot.any():
            d0, d1 = idx[1:, 1:][m_bot], idx[1:, :-1][m_bot]
            caras_costura.append(np.column_stack([d0, d1, d1 + N]))
            caras_costura.append(np.column_stack([d0, d1 + N, d0 + N]))

        # Aristas izquierdas (celda válida cuyo vecino izquierdo no es válido)
        m_left = valido & (~v_pad[1:-1, :-2])
        if m_left.any():
            l0, l1 = idx[1:, :-1][m_left], idx[:-1, :-1][m_left]
            caras_costura.append(np.column_stack([l0, l1, l1 + N]))
            caras_costura.append(np.column_stack([l0, l1 + N, l0 + N]))

        # Aristas derechas (celda válida cuyo vecino derecho no es válido)
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
    })
    return malla


# --------------------------------------------------------------------------------------
# Vistas desde otros ángulos (sin necesidad de ventana ni GPU) para verificar la geometría
# --------------------------------------------------------------------------------------
def renderizar_vista(malla: trimesh.Trimesh, yaw_grados: float = 0.0, pitch_grados: float = 0.0,
                     tam_punto: int = 2) -> np.ndarray:
    """Proyecta los vértices de la malla desde una cámara rotada alrededor del centro de la escena."""
    f, W, H = malla.metadata["f"], malla.metadata["W"], malla.metadata["H"]
    piv = np.array([0.0, 0.0, -malla.metadata["z_pivote"]])
    P = np.asarray(malla.vertices) - piv
    col = np.asarray(malla.visual.vertex_colors)[:, :3]

    ya, pa = np.deg2rad(yaw_grados), np.deg2rad(pitch_grados)
    Ry = np.array([[np.cos(ya), 0, np.sin(ya)], [0, 1, 0], [-np.sin(ya), 0, np.cos(ya)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pa), -np.sin(pa)], [0, np.sin(pa), np.cos(pa)]])
    P = (Rx @ Ry @ P.T).T + piv

    prof = -P[:, 2]
    ok = prof > 0.05
    P, col, prof = P[ok], col[ok], prof[ok]
    u = np.round(f * P[:, 0] / prof + (W - 1) / 2).astype(int)
    v = np.round(-f * P[:, 1] / prof + (H - 1) / 2).astype(int)

    lienzo = np.full((H, W, 3), 28, np.uint8)
    orden = np.argsort(-prof)                       # algoritmo del pintor: lejos -> cerca
    u, v, col = u[orden], v[orden], col[orden]
    for du in range(tam_punto):
        for dv in range(tam_punto):
            uu, vv = u + du, v + dv
            m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            lienzo[vv[m], uu[m]] = col[m]
    return lienzo


def guardar_vistas(malla: trimesh.Trimesh, ruta_png: str, yaws=(-45, -20, 0, 20, 45, 135, 180), pitch=0.0):
    tiras = [renderizar_vista(malla, y, pitch) for y in yaws]
    cv2.imwrite(ruta_png, cv2.cvtColor(np.hstack(tiras), cv2.COLOR_RGB2BGR))


def visualizar_y_exportar(malla: trimesh.Trimesh, ruta_obj: str = "data/modelo_3d.obj",
                          mostrar: bool = True):
    carpeta = os.path.dirname(ruta_obj) or "."
    os.makedirs(carpeta, exist_ok=True)
    base = os.path.splitext(ruta_obj)[0]
    malla.export(ruta_obj)                 # OBJ (colores por vértice como extensión)
    malla.export(base + ".ply")            # PLY: conserva bien el color en MeshLab/CloudCompare/Blender
    malla.export(base + ".glb")            # GLB: se abre directo en visores web y Windows 3D Viewer
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
    opt.light_on = False                    # el color ya viene de la foto; la luz lo "lavaba"
    ctr = vis.get_view_control()
    zp = malla.metadata.get("z_pivote", 2.0)
    ctr.set_lookat([0.0, 0.0, -zp])
    ctr.set_front([0.0, 0.0, 1.0])          # la cámara original estaba en +Z mirando hacia -Z
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_zoom(0.5)
    vis.run()
    vis.destroy_window()
