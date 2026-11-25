"""
Uso (desde la carpeta raíz del proyecto):
    python src/main.py                                        # Abre el Menú Principal interactivo
    python src/main.py --imagen data/prueba.jpg --sin-menu    # Ejecución directa por consola
    python src/main.py --imagen data/foto.png --fondo auto --relleno-espejo --grosor-espejo 0.45 --sin-menu
"""
import argparse
import os
import threading
import time

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageTk

from fase1_slic import (
    cargar_imagen_cualquier_formato,
    clasificar_tipo_escena,
    estimar_punto_fuga_hough,
    procesar_superpixeles,
    segmentar_objeto_sin_fondo,
)
from fase2_clustering_profundidad import agrupar_e_inferir_profundidad
from fase3_malla_3d import construir_malla_volumetrica, guardar_vistas, visualizar_y_exportar


def generar_imagen_slic(img_rgb: np.ndarray, slic_map: np.ndarray, mascara_fg=None) -> np.ndarray:
    """
    Genera una visualización clara de los superpíxeles SLIC:
      - Izquierda: fotografía original con la malla de fronteras de cada superpíxel SLIC.
      - Derecha: mosaico de color promedio por superpíxel (representación simplificada que alimenta a K-Means).
    """
    bordes = np.zeros(slic_map.shape, dtype=bool)
    bordes[:, 1:] |= (slic_map[:, 1:] != slic_map[:, :-1])
    bordes[1:, :] |= (slic_map[1:, :] != slic_map[:-1, :])

    # Panel izquierdo: imagen original + malla SLIC en amarillo dorado de alto contraste
    vis_izq = img_rgb.copy()
    if mascara_fg is not None:
        vis_izq[~mascara_fg] = (vis_izq[~mascara_fg] * 0.45).astype(np.uint8)
    vis_izq[bordes] = (255, 215, 60)

    # Panel derecho: color medio de cada superpíxel SLIC + bordes suaves
    lab = slic_map.ravel()
    n_sp = int(lab.max()) + 1
    cnt = np.bincount(lab, minlength=n_sp).astype(np.float64) + 1e-9
    colores_sp = np.zeros((n_sp, 3), dtype=np.uint8)
    for ch in range(3):
        medias = np.bincount(lab, weights=img_rgb[:, :, ch].ravel().astype(np.float64), minlength=n_sp) / cnt
        colores_sp[:, ch] = np.clip(medias, 0, 255).astype(np.uint8)
    vis_der = colores_sp[slic_map]
    if mascara_fg is not None:
        vis_der[~mascara_fg] = (vis_der[~mascara_fg] * 0.45).astype(np.uint8)
    vis_der[bordes] = (28, 30, 38)

    sep = np.full((img_rgb.shape[0], 4, 3), 24, dtype=np.uint8)
    return np.hstack([vis_izq, sep, vis_der])


def generar_imagen_kmeans(img_rgb: np.ndarray, mapa_k: np.ndarray, m: dict, mascara_fg=None) -> np.ndarray:
    """
    Genera una visualización clara de cómo K-Means dividió la imagen en K secciones:
      - Izquierda: imagen original superpuesta con los colores y contornos de cada clúster.
      - Derecha: mapa puro de secciones K-Means con bordes nítidos y etiquetas C0..CK-1.
    """
    paleta = np.array([
        [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
        [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
        [188, 189, 34], [23, 190, 207], [174, 199, 232], [255, 187, 120],
        [152, 223, 138], [255, 152, 150], [197, 176, 213], [196, 156, 148],
        [247, 182, 210], [199, 199, 199], [219, 219, 141], [158, 218, 229],
    ], dtype=np.uint8)

    col_k = paleta[mapa_k % len(paleta)]

    # Detectar fronteras entre secciones K-Means
    bordes = np.zeros(mapa_k.shape, dtype=bool)
    bordes[:, 1:] |= (mapa_k[:, 1:] != mapa_k[:, :-1])
    bordes[1:, :] |= (mapa_k[1:, :] != mapa_k[:-1, :])
    bordes = cv2.dilate(bordes.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0

    # Panel izquierdo: imagen original + tinte semitransparente de secciones + bordes
    vis_izq = (0.55 * img_rgb.astype(np.float32) + 0.45 * col_k.astype(np.float32)).astype(np.uint8)
    vis_izq[bordes] = (20, 22, 28)

    # Panel derecho: mapa de secciones puro + bordes + etiquetas C0..CK-1 en el centroide de cada clúster
    vis_der = col_k.copy()
    if mascara_fg is not None:
        vis_der[~mascara_fg] = (vis_der[~mascara_fg] * 0.45).astype(np.uint8)
    vis_der[bordes] = (18, 20, 26)

    escala_txt = max(0.45, min(img_rgb.shape[:2]) / 680.0)
    for cid in np.unique(mapa_k):
        mask_c = (mapa_k == cid)
        if mascara_fg is not None and (mask_c & mascara_fg).sum() > 40:
            mask_c = mask_c & mascara_fg
        num_cc, labels_cc, stats_cc, centroids_cc = cv2.connectedComponentsWithStats(
            mask_c.astype(np.uint8), connectivity=8
        )
        if num_cc <= 1:
            continue
        idx_may = 1 + int(np.argmax(stats_cc[1:, cv2.CC_STAT_AREA]))
        cx, cy = int(round(centroids_cc[idx_may][0])), int(round(centroids_cc[idx_may][1]))
        etiqueta = f"C{int(cid)}"
        cv2.putText(vis_der, etiqueta, (cx - 12, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, escala_txt, (15, 15, 15), 3, cv2.LINE_AA)
        cv2.putText(vis_der, etiqueta, (cx - 12, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, escala_txt, (255, 255, 255), 1, cv2.LINE_AA)

    sep = np.full((img_rgb.shape[0], 4, 3), 24, dtype=np.uint8)
    return np.hstack([vis_izq, sep, vis_der]), vis_der


def guardar_diagnostico(ruta_png, img_rgb, mapa_k, z_bruto, z_final, m, mascara_fg=None, vis_kmeans=None):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
    H, W = img_rgb.shape[:2]

    if mascara_fg is not None:
        vis_fg = img_rgb.copy()
        vis_fg[~mascara_fg] = (vis_fg[~mascara_fg] * 0.25).astype(np.uint8)
        contornos, _ = cv2.findContours(mascara_fg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_fg, contornos, -1, (0, 255, 120), 2)
        ax[0].imshow(vis_fg)
        ax[0].set_title(f"1. Objeto segmentado [{m.get('tipo_escena', 'objeto')}]")
    else:
        ax[0].imshow(img_rgb)
        ax[0].axhline(m["y_horizonte"] * H, color="lime", ls="--", label="Horizonte")
        ax[0].plot(m["x_vp"] * W, m["y_horizonte"] * H, "ro", label=f"Punto fuga (conf {m['vp_confianza']})")
        ax[0].legend(loc="upper right", fontsize=7)
        ax[0].set_title(f"1. Escena completa [{m.get('tipo_escena', 'escena')}]")

    if vis_kmeans is not None:
        ax[1].imshow(vis_kmeans)
    else:
        ax[1].imshow(mapa_k, cmap="tab10", vmin=0, vmax=9)
    ax[1].set_title(f"2. K-Means (K={m['k_clusters']})")
    ax[2].imshow(z_bruto, cmap="magma")
    ax[2].set_title("3. Profundidad (planos por superpíxel)")
    ax[3].imshow(z_final, cmap="magma")
    etiqueta_espejo = f" + Espejo ({int(m.get('grosor_espejo', 0.65) * 100)}%)" if m.get("relleno_espejo", False) else ""
    ax[3].set_title(f"4. Profundidad final [{m['modo_profundidad']}{etiqueta_espejo}]")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    fig.savefig(ruta_png, dpi=110)
    plt.close(fig)


def ejecutar_pipeline(
    ruta_imagen: str,
    carpeta_salida: str = "data/salida",
    modo_fondo: str | bool = "auto",
    relleno_espejo: bool = True,
    grosor_espejo: float = 0.65,
    fov: float = 60.0,
    k: int = 8,
    z_lejos: float = 8.0,
    umbral: float = 0.20,
    max_dim: int = 640,
    mostrar_visor: bool = True,
    callback_progreso=None,
    callback_etapa=None,
):
    def informar(msg: str, pct: int):
        print(f"[{pct:3d}%] {msg}")
        if callback_progreso is not None:
            callback_progreso(msg, pct)

    os.makedirs(carpeta_salida, exist_ok=True)
    t0 = time.time()

    informar("Cargando imagen, analizando tipo de escena y superpíxeles SLIC...", 15)
    img_rgb, slic_map, feats, info = procesar_superpixeles(
        ruta_imagen, max_dim=max_dim, modo_fondo=modo_fondo
    )
    mascara_fg = info.get("mascara_fg", None)
    tipo_escena = info.get("tipo_escena", "escena")

    img_slic = generar_imagen_slic(img_rgb, slic_map, mascara_fg=mascara_fg)
    ruta_slic = os.path.join(carpeta_salida, "slic_superpixeles.png")
    cv2.imwrite(ruta_slic, cv2.cvtColor(img_slic, cv2.COLOR_RGB2BGR))
    if callback_etapa is not None:
        callback_etapa(
            "slic",
            img_slic,
            f"SLIC ({len(feats)} superpíxeles): Malla sobre imagen (Izq.) | Color promedio por celda (Der.)",
        )
    informar(f"SLIC completado ({len(feats)} superpíxeles). Aplicando K-Means (K={k})...", 45)
    mapa_k, z_bruto, z_final, m = agrupar_e_inferir_profundidad(
        img_rgb, slic_map, feats, info, n_clusters=k
    )
    m["relleno_espejo"] = bool(relleno_espejo)
    m["grosor_espejo"] = round(float(grosor_espejo), 2)

    img_kmeans, vis_kmeans_der = generar_imagen_kmeans(img_rgb, mapa_k, m, mascara_fg=mascara_fg)
    ruta_kmeans = os.path.join(carpeta_salida, "kmeans_secciones.png")
    cv2.imwrite(ruta_kmeans, cv2.cvtColor(img_kmeans, cv2.COLOR_RGB2BGR))
    if callback_etapa is not None:
        callback_etapa(
            "kmeans",
            img_kmeans,
            f"K-Means (K={m['k_clusters']}) — Secciones | Silueta: {m['silueta']} | DBI: {m['davies_bouldin']}",
        )
    informar(f"K-Means aplicado ({m['k_clusters']} secciones, Silueta={m['silueta']})...", 68)

    print("\n=== MÉTRICAS Y GEOMETRÍA DE ESCENA ===")
    for clave, valor in m.items():
        print(f"• {clave:22s}: {valor}")

    ruta_diag = os.path.join(carpeta_salida, "diagnostico.png")
    ruta_prof = os.path.join(carpeta_salida, "profundidad.png")
    guardar_diagnostico(ruta_diag, img_rgb, mapa_k, z_bruto, z_final, m, mascara_fg=mascara_fg, vis_kmeans=vis_kmeans_der)
    cv2.imwrite(ruta_prof, (z_final * 255).astype(np.uint8))

    if mascara_fg is not None:
        rgba_sin_fondo = np.dstack([img_rgb, (mascara_fg.astype(np.uint8) * 255)])
        Image.fromarray(rgba_sin_fondo).save(os.path.join(carpeta_salida, "sin_fondo.png"))

    msg_espejo = f" con relleno espejo (grosor {int(grosor_espejo * 100)}%)..." if relleno_espejo else "..."
    informar("Construyendo malla 3D" + msg_espejo, 85)
    malla = construir_malla_volumetrica(
        img_rgb,
        z_final,
        mapa_k,
        m["y_horizonte"],
        fov_grados=fov,
        z_lejos=z_lejos,
        umbral_discontinuidad=umbral,
        mascara_fg=mascara_fg,
        relleno_espejo=relleno_espejo,
        grosor_espejo=grosor_espejo,
        mapa_cavidad=info.get("mapa_cavidad", None),
        mapa_realce=info.get("mapa_realce", None),
        factor_plano=info.get("factor_plano", 0.0),
        es_horizontal_34=info.get("es_horizontal_34", False),
    )

    ruta_vistas = os.path.join(carpeta_salida, "vistas.png")
    ruta_obj = os.path.join(carpeta_salida, "modelo_3d.obj")
    guardar_vistas(malla, ruta_vistas)
    visualizar_y_exportar(malla, ruta_obj, mostrar=False)

    dt = time.time() - t0
    informar(f"¡Modelo 3D completado en {dt:.1f} s! ({len(malla.vertices):,} vértices, {len(malla.faces):,} caras)", 100)

    if mostrar_visor:
        visualizar_y_exportar(malla, ruta_obj, mostrar=True)

    return {
        "malla": malla,
        "metricas": m,
        "ruta_slic": ruta_slic,
        "ruta_kmeans": ruta_kmeans,
        "ruta_diag": ruta_diag,
        "ruta_vistas": ruta_vistas,
        "ruta_obj": ruta_obj,
        "tiempo": dt,
    }


def abrir_menu_principal(args_iniciales):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Generador Universal de Modelos 3D - Menú Principal")
    root.geometry("1100x760")
    root.minsize(960, 680)
    root.configure(bg="#181920")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    BG_MAIN = "#181920"
    BG_CARD = "#232530"
    FG_TEXT = "#F2F4F8"
    FG_SUB = "#AAB0C0"
    ACCENT = "#3B82F6"
    ACCENT_HOVER = "#2563EB"

    style.configure("TFrame", background=BG_MAIN)
    style.configure("Card.TFrame", background=BG_CARD)
    style.configure("TLabel", background=BG_CARD, foreground=FG_TEXT, font=("Segoe UI", 10))
    style.configure("Header.TLabel", background=BG_MAIN, foreground=FG_TEXT, font=("Segoe UI", 16, "bold"))
    style.configure("SubHeader.TLabel", background=BG_MAIN, foreground=FG_SUB, font=("Segoe UI", 10))
    style.configure("Section.TLabel", background=BG_CARD, foreground="#60A5FA", font=("Segoe UI", 11, "bold"))
    style.configure("TCheckbutton", background=BG_CARD, foreground=FG_TEXT, font=("Segoe UI", 10))
    style.map("TCheckbutton", background=[("active", BG_CARD)])
    style.configure("TRadiobutton", background=BG_CARD, foreground=FG_TEXT, font=("Segoe UI", 10))
    style.map("TRadiobutton", background=[("active", BG_CARD)])
    style.configure("Horizontal.TScale", background=BG_CARD)
    style.configure(
        "Primary.TButton",
        background=ACCENT,
        foreground="#FFFFFF",
        font=("Segoe UI", 11, "bold"),
        padding=(14, 10),
    )
    style.map("Primary.TButton", background=[("active", ACCENT_HOVER), ("disabled", "#4B5563")])
    style.configure(
        "Secondary.TButton",
        background="#343846",
        foreground=FG_TEXT,
        font=("Segoe UI", 9, "bold"),
        padding=(8, 5),
    )
    style.map("Secondary.TButton", background=[("active", "#454A5C")])

    # Variables de control
    var_imagen = tk.StringVar(value=args_iniciales.imagen if os.path.exists(args_iniciales.imagen) else "")
    var_salida = tk.StringVar(value=args_iniciales.salida)
    var_modo_fondo = tk.StringVar(value=args_iniciales.fondo)
    var_relleno_espejo = tk.BooleanVar(value=not args_iniciales.sin_espejo)
    var_grosor_espejo = tk.DoubleVar(value=args_iniciales.grosor_espejo)
    var_fov = tk.DoubleVar(value=args_iniciales.fov)
    var_k = tk.IntVar(value=args_iniciales.k)
    var_max_dim = tk.IntVar(value=args_iniciales.max_dim)
    var_abrir_visor = tk.BooleanVar(value=not args_iniciales.sin_visor)
    var_estado = tk.StringVar(value="Selecciona una imagen (objeto, auto, casa o paisaje) para generar el modelo 3D.")

    # Encabezado
    header_frame = ttk.Frame(root)
    header_frame.pack(fill="x", padx=20, pady=(12, 6))
    ttk.Label(header_frame, text="Generador Universal de Modelos 3D", style="Header.TLabel").pack(anchor="w")
    ttk.Label(
        header_frame,
        text="Modela cualquier imagen (objetos, autos, casas, paisajes) con alta distinción de profundidad, opción de quitar el fondo y control de grosor en modo espejo.",
        style="SubHeader.TLabel",
    ).pack(anchor="w", pady=(2, 0))

    # Contenedor central
    main_frame = ttk.Frame(root)
    main_frame.pack(fill="both", expand=True, padx=20, pady=6)
    main_frame.columnconfigure(0, weight=0, minsize=405)
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(0, weight=1)

    # Panel izquierdo
    panel_izq = ttk.Frame(main_frame, style="Card.TFrame", padding=15)
    panel_izq.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

    ttk.Label(panel_izq, text="1. Selección de Fotografía", style="Section.TLabel").pack(anchor="w", pady=(0, 5))

    fila_img = ttk.Frame(panel_izq, style="Card.TFrame")
    fila_img.pack(fill="x", pady=(0, 4))
    entry_img = ttk.Entry(fila_img, textvariable=var_imagen, font=("Segoe UI", 9))
    entry_img.pack(side="left", fill="x", expand=True, padx=(0, 6))

    def elegir_archivo():
        ruta = filedialog.askopenfilename(
            title="Seleccionar fotografía",
            initialdir=os.path.dirname(var_imagen.get()) if var_imagen.get() else "data",
            filetypes=[
                ("Imágenes soportadas", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff;*.gif"),
                ("PNG", "*.png"),
                ("JPEG / JPG", "*.jpg;*.jpeg"),
                ("WEBP / BMP / TIFF", "*.webp;*.bmp;*.tif;*.tiff"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if ruta:
            var_imagen.set(ruta)
            actualizar_vista_previa()

    ttk.Button(fila_img, text="Examinar...", style="Secondary.TButton", command=elegir_archivo).pack(side="right")

    ttk.Separator(panel_izq, orient="horizontal").pack(fill="x", pady=8)

    ttk.Label(panel_izq, text="2. Manejo del Fondo y Modo Espejo 3D", style="Section.TLabel").pack(anchor="w", pady=(0, 5))

    ttk.Radiobutton(
        panel_izq,
        text="Automático inteligente (aísla objetos/autos; conserva paisajes)",
        variable=var_modo_fondo,
        value="auto",
        command=lambda: actualizar_vista_previa(),
    ).pack(anchor="w", pady=2)

    ttk.Radiobutton(
        panel_izq,
        text="Quitar el fondo (aislar objeto / vehículo / figura)",
        variable=var_modo_fondo,
        value="quitar",
        command=lambda: actualizar_vista_previa(),
    ).pack(anchor="w", pady=2)

    ttk.Radiobutton(
        panel_izq,
        text="Conservar el fondo (modelar paisaje / calle / casa completa)",
        variable=var_modo_fondo,
        value="conservar",
        command=lambda: actualizar_vista_previa(),
    ).pack(anchor="w", pady=(2, 6))

    chk_espejo = ttk.Checkbutton(
        panel_izq,
        text="Relleno automático trasero (duplicar en espejo 3D)",
        variable=var_relleno_espejo,
    )
    chk_espejo.pack(anchor="w", pady=2)

    # Control de grosor del modo espejo (con deslizador y botones rápidos para hacerlo menos grueso)
    frame_grosor = ttk.Frame(panel_izq, style="Card.TFrame")
    frame_grosor.pack(fill="x", pady=(4, 4))

    fila_lbl_grosor = ttk.Frame(frame_grosor, style="Card.TFrame")
    fila_lbl_grosor.pack(fill="x")
    ttk.Label(fila_lbl_grosor, text="Grosor del modelo en espejo:").pack(side="left")
    lbl_val_grosor = ttk.Label(fila_lbl_grosor, text="", foreground="#60A5FA", font=("Segoe UI", 9, "bold"))
    lbl_val_grosor.pack(side="right")

    def actualizar_etiqueta_grosor(*_):
        v = float(var_grosor_espejo.get())
        pct = int(round(v * 100))
        if v <= 0.30:
            desc = "Muy delgado"
        elif v <= 0.52:
            desc = "Delgado"
        elif v <= 0.80:
            desc = "Equilibrado"
        elif v <= 1.10:
            desc = "Normal"
        else:
            desc = "Grueso"
        lbl_val_grosor.config(text=f"{pct}% ({desc})")

    scale_grosor = ttk.Scale(
        frame_grosor,
        from_=0.15,
        to=1.40,
        orient="horizontal",
        variable=var_grosor_espejo,
        command=actualizar_etiqueta_grosor,
    )
    scale_grosor.pack(fill="x", pady=(3, 4))

    fila_presets_grosor = ttk.Frame(frame_grosor, style="Card.TFrame")
    fila_presets_grosor.pack(fill="x")

    def fijar_grosor(val: float):
        var_grosor_espejo.set(val)
        actualizar_etiqueta_grosor()

    ttk.Button(
        fila_presets_grosor, text="Muy delgado (25%)", style="Secondary.TButton",
        command=lambda: fijar_grosor(0.25)
    ).pack(side="left", expand=True, fill="x", padx=(0, 3))
    ttk.Button(
        fila_presets_grosor, text="Delgado (45%)", style="Secondary.TButton",
        command=lambda: fijar_grosor(0.45)
    ).pack(side="left", expand=True, fill="x", padx=3)
    ttk.Button(
        fila_presets_grosor, text="Equilibrado (65%)", style="Secondary.TButton",
        command=lambda: fijar_grosor(0.65)
    ).pack(side="left", expand=True, fill="x", padx=3)
    ttk.Button(
        fila_presets_grosor, text="Normal (100%)", style="Secondary.TButton",
        command=lambda: fijar_grosor(1.00)
    ).pack(side="left", expand=True, fill="x", padx=(3, 0))

    actualizar_etiqueta_grosor()

    def al_cambiar_espejo():
        estado_s = "normal" if var_relleno_espejo.get() else "disabled"
        scale_grosor.configure(state=estado_s)

    chk_espejo.configure(command=al_cambiar_espejo)

    chk_visor = ttk.Checkbutton(
        panel_izq,
        text="Abrir visor 3D interactivo (Open3D) al terminar",
        variable=var_abrir_visor,
    )
    chk_visor.pack(anchor="w", pady=(4, 2))

    ttk.Separator(panel_izq, orient="horizontal").pack(fill="x", pady=8)

    ttk.Label(panel_izq, text="3. Parámetros de Reconstrucción", style="Section.TLabel").pack(anchor="w", pady=(0, 5))

    grid_params = ttk.Frame(panel_izq, style="Card.TFrame")
    grid_params.pack(fill="x")
    grid_params.columnconfigure(1, weight=1)

    ttk.Label(grid_params, text="Clusters K-Means (K):").grid(row=0, column=0, sticky="w", pady=2, padx=(0, 8))
    spin_k = ttk.Spinbox(grid_params, from_=3, to=20, textvariable=var_k, width=10)
    spin_k.grid(row=0, column=1, sticky="ew", pady=2)

    ttk.Label(grid_params, text="Campo de visión (FOV°):").grid(row=1, column=0, sticky="w", pady=2, padx=(0, 8))
    spin_fov = ttk.Spinbox(grid_params, from_=35.0, to=100.0, increment=5.0, textvariable=var_fov, width=10)
    spin_fov.grid(row=1, column=1, sticky="ew", pady=2)

    ttk.Label(grid_params, text="Resolución máx. (px):").grid(row=2, column=0, sticky="w", pady=2, padx=(0, 8))
    spin_dim = ttk.Spinbox(grid_params, from_=320, to=1024, increment=80, textvariable=var_max_dim, width=10)
    spin_dim.grid(row=2, column=1, sticky="ew", pady=2)

    ttk.Label(grid_params, text="Carpeta de salida:").grid(row=3, column=0, sticky="w", pady=2, padx=(0, 8))
    entry_sal = ttk.Entry(grid_params, textvariable=var_salida, font=("Segoe UI", 9))
    entry_sal.grid(row=3, column=1, sticky="ew", pady=2)

    ttk.Separator(panel_izq, orient="horizontal").pack(fill="x", pady=10)

    btn_generar = ttk.Button(panel_izq, text="Generar Modelo 3D", style="Primary.TButton")
    btn_generar.pack(fill="x", pady=(2, 6))

    barra_progreso = ttk.Progressbar(panel_izq, orient="horizontal", mode="determinate", maximum=100)
    barra_progreso.pack(fill="x", pady=(0, 5))

    lbl_estado = ttk.Label(
        panel_izq,
        textvariable=var_estado,
        wraplength=370,
        foreground=FG_SUB,
        font=("Segoe UI", 9),
    )
    lbl_estado.pack(anchor="w")

    # Panel derecho: Vista previa
    panel_der = ttk.Frame(main_frame, style="Card.TFrame", padding=16)
    panel_der.grid(row=0, column=1, sticky="nsew")
    panel_der.rowconfigure(1, weight=1)
    panel_der.columnconfigure(0, weight=1)

    header_der = ttk.Frame(panel_der, style="Card.TFrame")
    header_der.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    lbl_titulo_previa = ttk.Label(
        header_der,
        text="Vista Previa",
        style="Section.TLabel",
    )
    lbl_titulo_previa.pack(side="left")

    botones_der = ttk.Frame(header_der, style="Card.TFrame")
    botones_der.pack(side="right")

    estado_preview = {"photo": None, "ultimo_res": None}

    def mostrar_imagen_archivo(clave_ruta: str, titulo: str):
        res_act = estado_preview.get("ultimo_res")
        if not res_act:
            return
        ruta_f = res_act.get(clave_ruta, "")
        if ruta_f and os.path.exists(ruta_f):
            bgr = cv2.imread(ruta_f)
            if bgr is not None:
                mostrar_array_en_canvas(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), titulo)

    btn_ver_slic = ttk.Button(
        botones_der,
        text="SLIC",
        style="Secondary.TButton",
        state="disabled",
        command=lambda: mostrar_imagen_archivo(
            "ruta_slic",
            f"SLIC ({estado_preview['ultimo_res']['metricas']['n_superpixeles']} superpíxeles): Malla (Izq.) | Mosaico (Der.)",
        ),
    )
    btn_ver_slic.pack(side="left", padx=(0, 4))

    btn_ver_kmeans = ttk.Button(
        botones_der,
        text="Secciones K-Means",
        style="Secondary.TButton",
        state="disabled",
        command=lambda: mostrar_imagen_archivo(
            "ruta_kmeans",
            f"K-Means (K={estado_preview['ultimo_res']['metricas']['k_clusters']}): División en secciones",
        ),
    )
    btn_ver_kmeans.pack(side="left", padx=(0, 4))

    btn_ver_diag = ttk.Button(
        botones_der,
        text="Diagnóstico",
        style="Secondary.TButton",
        state="disabled",
        command=lambda: mostrar_imagen_archivo("ruta_diag", "Diagnóstico Completo (4 Fases)"),
    )
    btn_ver_diag.pack(side="left", padx=(0, 4))

    btn_ver_vistas = ttk.Button(
        botones_der,
        text="Vistas 3D",
        style="Secondary.TButton",
        state="disabled",
        command=lambda: mostrar_imagen_archivo(
            "ruta_vistas",
            f"Resultado 3D: Vistas Multi-ángulo (Grosor Espejo: {int(var_grosor_espejo.get() * 100)}%)",
        ),
    )
    btn_ver_vistas.pack(side="left", padx=(0, 4))

    ttk.Button(
        botones_der,
        text="Vista previa",
        style="Secondary.TButton",
        command=lambda: actualizar_vista_previa(),
    ).pack(side="left")

    canvas_previa = tk.Label(panel_der, bg="#121318", fg=FG_SUB, text="Sin imagen seleccionada", font=("Segoe UI", 11))
    canvas_previa.grid(row=1, column=0, sticky="nsew")

    def mostrar_array_en_canvas(arr_rgb: np.ndarray, titulo: str | None = None):
        if titulo:
            lbl_titulo_previa.config(text=titulo)
        cw = max(canvas_previa.winfo_width(), 540)
        ch = max(canvas_previa.winfo_height(), 420)
        ih, iw = arr_rgb.shape[:2]
        escala = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * escala)), max(1, int(ih * escala))
        pil_resized = Image.fromarray(arr_rgb).resize((nw, nh), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil_resized)
        estado_preview["photo"] = photo
        canvas_previa.config(image=photo, text="")

    def actualizar_vista_previa():
        ruta = var_imagen.get().strip()
        if not ruta or not os.path.exists(ruta):
            canvas_previa.config(image="", text="Selecciona una imagen válida para ver la vista previa.")
            return
        try:
            img_bgr, mascara_alfa = cargar_imagen_cualquier_formato(ruta, max_dim=440)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _, y_h, vp_conf = estimar_punto_fuga_hough(gris)
            tipo_esc, rec_quitar = clasificar_tipo_escena(img_bgr, mascara_alfa, vp_conf, y_h)

            mf = var_modo_fondo.get()
            quitar = rec_quitar if mf == "auto" else (mf == "quitar")

            if quitar or mascara_alfa is not None:
                fg = segmentar_objeto_sin_fondo(img_bgr, mascara_alfa)
                h, w = img_rgb.shape[:2]
                yy, xx = np.mgrid[0:h, 0:w]
                damero = (((yy // 16) + (xx // 16)) % 2 == 0)
                bg_chess = np.where(damero[:, :, None], 56, 36).astype(np.uint8)
                recorte = np.where(fg[:, :, None], img_rgb, bg_chess)
                combinada = np.hstack([img_rgb, recorte])
                mostrar_array_en_canvas(
                    combinada,
                    f"Vista Previa [{tipo_esc.upper()}]: Original (Izq.) | Objeto Aislado sin Fondo (Der.)",
                )
            else:
                mostrar_array_en_canvas(
                    img_rgb,
                    f"Vista Previa [{tipo_esc.upper()}]: Escena Completa (se modelará terreno, estructuras y fondo)",
                )
        except Exception as e:
            canvas_previa.config(image="", text=f"Error al previsualizar: {e}")

    def iniciar_generacion():
        ruta = var_imagen.get().strip()
        if not ruta or not os.path.exists(ruta):
            messagebox.showerror("Archivo no encontrado", "Por favor escoge una fotografía válida antes de generar el modelo 3D.")
            return

        btn_generar.config(state="disabled")
        barra_progreso["value"] = 5

        def callback_ui(msg, pct):
            root.after(0, lambda: (var_estado.set(msg), barra_progreso.configure(value=pct)))

        def callback_etapa_ui(_nombre, arr_rgb, titulo):
            copia = arr_rgb.copy()
            root.after(0, lambda: mostrar_array_en_canvas(copia, titulo))

        def tarea_fondo():
            try:
                res = ejecutar_pipeline(
                    ruta_imagen=ruta,
                    carpeta_salida=var_salida.get().strip() or "data/salida",
                    modo_fondo=var_modo_fondo.get(),
                    relleno_espejo=var_relleno_espejo.get(),
                    grosor_espejo=float(var_grosor_espejo.get()),
                    fov=float(var_fov.get()),
                    k=int(var_k.get()),
                    max_dim=int(var_max_dim.get()),
                    mostrar_visor=False,
                    callback_progreso=callback_ui,
                    callback_etapa=callback_etapa_ui,
                )

                def al_terminar():
                    btn_generar.config(state="normal")
                    estado_preview["ultimo_res"] = res
                    btn_ver_slic.config(state="normal")
                    btn_ver_kmeans.config(state="normal")
                    btn_ver_diag.config(state="normal")
                    btn_ver_vistas.config(state="normal")
                    if os.path.exists(res["ruta_vistas"]):
                        vistas_bgr = cv2.imread(res["ruta_vistas"])
                        if vistas_bgr is not None:
                            vistas_rgb = cv2.cvtColor(vistas_bgr, cv2.COLOR_BGR2RGB)
                            mostrar_array_en_canvas(
                                vistas_rgb,
                                f"Resultado 3D: Vistas Multi-ángulo (Grosor Espejo: {int(var_grosor_espejo.get() * 100)}%)",
                            )
                    if var_abrir_visor.get():
                        threading.Thread(
                            target=visualizar_y_exportar,
                            args=(res["malla"], res["ruta_obj"], True),
                            daemon=True,
                        ).start()

                root.after(0, al_terminar)
            except Exception as e:
                def al_fallar():
                    btn_generar.config(state="normal")
                    var_estado.set(f"Error: {e}")
                    messagebox.showerror("Error en la generación 3D", str(e))
                root.after(0, al_fallar)

        threading.Thread(target=tarea_fondo, daemon=True).start()

    btn_generar.config(command=iniciar_generacion)
    root.after(150, actualizar_vista_previa)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Generador universal de modelos 3D a partir de cualquier imagen.")
    ap.add_argument("--imagen", default="data/prueba.jpg", help="Ruta de la foto (PNG, JPG, WEBP, BMP, etc.)")
    ap.add_argument("--salida", default="data/salida", help="Carpeta donde se exportará el modelo 3D")
    ap.add_argument("--fondo", choices=["auto", "quitar", "conservar"], default="auto",
                    help="Modo de fondo: 'auto' (según escena), 'quitar' (aislar objeto) o 'conservar' (escena completa)")
    ap.add_argument("--quitar-fondo", action="store_true", help="Fuerza quitar el fondo de la foto")
    ap.add_argument("--conservar-fondo", action="store_true", help="Fuerza conservar el fondo original de la escena")
    ap.add_argument("--fov", type=float, default=60.0, help="Campo de visión horizontal (grados)")
    ap.add_argument("--k", type=int, default=8, help="Número de clusters K-Means")
    ap.add_argument("--z-lejos", type=float, default=8.0, help="Profundidad del infinito relativa a lo más cercano (=1)")
    ap.add_argument("--umbral", type=float, default=0.20, help="Corte de malla en saltos de profundidad")
    ap.add_argument("--max-dim", type=int, default=640, help="Dimensión máxima de procesamiento en píxeles")
    ap.add_argument("--relleno-espejo", action="store_true", default=True, help="Duplicar el modelo por detrás en espejo (activo por defecto)")
    ap.add_argument("--sin-espejo", action="store_true", help="Generar solo la superficie frontal sin relleno trasero")
    ap.add_argument("--grosor-espejo", type=float, default=0.65,
                    help="Factor de grosor al duplicar en espejo (ej. 0.25 = muy delgado, 0.45 = delgado, 0.65 = equilibrado, 1.0 = normal)")
    ap.add_argument("--sin-visor", action="store_true", help="No abrir la ventana de Open3D al finalizar")
    ap.add_argument("--sin-menu", action="store_true", help="Ejecutar directamente por consola sin abrir el menú principal")
    args = ap.parse_args()

    if args.quitar_fondo:
        args.fondo = "quitar"
    elif args.conservar_fondo:
        args.fondo = "conservar"

    if not args.sin_menu:
        abrir_menu_principal(args)
        return

    espejo = not args.sin_espejo

    ejecutar_pipeline(
        ruta_imagen=args.imagen,
        carpeta_salida=args.salida,
        modo_fondo=args.fondo,
        relleno_espejo=espejo,
        grosor_espejo=args.grosor_espejo,
        fov=args.fov,
        k=args.k,
        z_lejos=args.z_lejos,
        umbral=args.umbral,
        max_dim=args.max_dim,
        mostrar_visor=not args.sin_visor,
    )


if __name__ == "__main__":
    main()
