"""
Uso (desde la carpeta raíz del proyecto):
    python src/main.py                                        # Abre el Menú Principal interactivo
    python src/main.py --imagen data/prueba.jpg --sin-menu    # Ejecución directa por consola
    python src/main.py --imagen data/foto.png --quitar-fondo --relleno-espejo --sin-menu
"""
import argparse
import os
import sys
import threading
import time

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageTk

from fase1_slic import (
    cargar_imagen_cualquier_formato,
    procesar_superpixeles,
    segmentar_objeto_sin_fondo,
)
from fase2_clustering_profundidad import agrupar_e_inferir_profundidad
from fase3_malla_3d import construir_malla_volumetrica, guardar_vistas, visualizar_y_exportar


def cargar_profundidad_neural(img_rgb, modo):
    if modo == "geometrico":
        return None
    try:
        from fase0_profundidad_neural import EstimadorProfundidadNeural
        return EstimadorProfundidadNeural().estimar(img_rgb)
    except Exception as e:
        if modo == "neural":
            raise
        print(f"[!] Red de profundidad no disponible ({type(e).__name__}: {e}). Uso modo geométrico.")
        return None


def guardar_diagnostico(ruta_png, img_rgb, mapa_k, z_bruto, z_final, m, mascara_fg=None):
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
        ax[0].set_title("1. Objeto segmentado (sin fondo)")
    else:
        ax[0].imshow(img_rgb)
        ax[0].axhline(m["y_horizonte"] * H, color="lime", ls="--", label="Horizonte")
        ax[0].plot(m["x_vp"] * W, m["y_horizonte"] * H, "ro", label=f"Punto fuga (conf {m['vp_confianza']})")
        ax[0].legend(loc="upper right", fontsize=7)
        ax[0].set_title("1. Escena + punto de fuga")

    ax[1].imshow(mapa_k, cmap="tab10")
    ax[1].set_title(f"2. K-Means (K={m['k_clusters']})")
    ax[2].imshow(z_bruto, cmap="magma")
    ax[2].set_title("3. Profundidad (planos por superpíxel)")
    ax[3].imshow(z_final, cmap="magma")
    etiqueta_espejo = " + Espejo 3D" if m.get("relleno_espejo", False) else ""
    ax[3].set_title(f"4. Profundidad final [{m['modo_profundidad']}{etiqueta_espejo}]")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    fig.savefig(ruta_png, dpi=110)
    plt.close(fig)


def ejecutar_pipeline(
    ruta_imagen: str,
    carpeta_salida: str = "data/salida",
    modo: str = "auto",
    quitar_fondo: bool = True,
    relleno_espejo: bool = True,
    fov: float = 60.0,
    k: int = 8,
    z_lejos: float = 8.0,
    umbral: float = 0.20,
    max_dim: int = 640,
    mostrar_visor: bool = True,
    callback_progreso=None,
):
    def informar(msg: str, pct: int):
        print(f"[{pct:3d}%] {msg}")
        if callback_progreso is not None:
            callback_progreso(msg, pct)

    os.makedirs(carpeta_salida, exist_ok=True)
    t0 = time.time()

    informar("Cargando imagen, segmentando fondo y calculando superpíxeles SLIC...", 15)
    img_rgb, slic_map, feats, info = procesar_superpixeles(
        ruta_imagen, max_dim=max_dim, quitar_fondo=quitar_fondo
    )
    mascara_fg = info.get("mascara_fg", None)

    informar("Estimando mapa de profundidad inicial...", 40)
    disp = cargar_profundidad_neural(img_rgb, modo)

    informar("Agrupando superpíxeles (K-Means) y refinando relieve volumétrico...", 65)
    mapa_k, z_bruto, z_final, m = agrupar_e_inferir_profundidad(
        img_rgb, slic_map, feats, info, n_clusters=k, disp_neural=disp
    )
    m["relleno_espejo"] = bool(relleno_espejo)

    print("\n=== MÉTRICAS Y GEOMETRÍA DE ESCENA ===")
    for clave, valor in m.items():
        print(f"• {clave:22s}: {valor}")

    ruta_diag = os.path.join(carpeta_salida, "diagnostico.png")
    ruta_prof = os.path.join(carpeta_salida, "profundidad.png")
    guardar_diagnostico(ruta_diag, img_rgb, mapa_k, z_bruto, z_final, m, mascara_fg=mascara_fg)
    cv2.imwrite(ruta_prof, (z_final * 255).astype(np.uint8))

    if mascara_fg is not None:
        rgba_sin_fondo = np.dstack([img_rgb, (mascara_fg.astype(np.uint8) * 255)])
        Image.fromarray(rgba_sin_fondo).save(os.path.join(carpeta_salida, "sin_fondo.png"))

    informar("Construyendo malla 3D" + (" con relleno espejo trasero..." if relleno_espejo else "..."), 85)
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
        "ruta_diag": ruta_diag,
        "ruta_vistas": ruta_vistas,
        "ruta_obj": ruta_obj,
        "tiempo": dt,
    }


def abrir_menu_principal(args_iniciales):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Generador de Modelos 3D - Menú Principal")
    root.geometry("1040x700")
    root.minsize(920, 620)
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
        padding=(10, 6),
    )
    style.map("Secondary.TButton", background=[("active", "#454A5C")])

    # Variables de control
    var_imagen = tk.StringVar(value=args_iniciales.imagen if os.path.exists(args_iniciales.imagen) else "")
    var_salida = tk.StringVar(value=args_iniciales.salida)
    var_quitar_fondo = tk.BooleanVar(value=True if not args_iniciales.conservar_fondo else False)
    var_relleno_espejo = tk.BooleanVar(value=True if not args_iniciales.sin_espejo else False)
    var_modo = tk.StringVar(value=args_iniciales.modo)
    var_fov = tk.DoubleVar(value=args_iniciales.fov)
    var_k = tk.IntVar(value=args_iniciales.k)
    var_max_dim = tk.IntVar(value=args_iniciales.max_dim)
    var_abrir_visor = tk.BooleanVar(value=not args_iniciales.sin_visor)
    var_estado = tk.StringVar(value="Selecciona una imagen y configura las opciones para generar el modelo 3D.")

    # Encabezado
    header_frame = ttk.Frame(root)
    header_frame.pack(fill="x", padx=20, pady=(16, 8))
    ttk.Label(header_frame, text="Generador de Modelos 3D Volumétricos", style="Header.TLabel").pack(anchor="w")
    ttk.Label(
        header_frame,
        text="Carga fotografías en cualquier formato (PNG, JPG, WEBP, BMP, TIFF), elimina el fondo y genera modelos 3D cerrados con relleno en espejo.",
        style="SubHeader.TLabel",
    ).pack(anchor="w", pady=(2, 0))

    # Contenedor central (Izquierda: Opciones | Derecha: Previsualización)
    main_frame = ttk.Frame(root)
    main_frame.pack(fill="both", expand=True, padx=20, pady=8)
    main_frame.columnconfigure(0, weight=0, minsize=380)
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(0, weight=1)

    # Panel izquierdo
    panel_izq = ttk.Frame(main_frame, style="Card.TFrame", padding=16)
    panel_izq.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

    ttk.Label(panel_izq, text="1. Selección de Fotografía", style="Section.TLabel").pack(anchor="w", pady=(0, 8))

    fila_img = ttk.Frame(panel_izq, style="Card.TFrame")
    fila_img.pack(fill="x", pady=(0, 6))
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

    ttk.Separator(panel_izq, orient="horizontal").pack(fill="x", pady=12)

    ttk.Label(panel_izq, text="2. Opciones de Geometría y Fondo", style="Section.TLabel").pack(anchor="w", pady=(0, 8))

    chk_fondo = ttk.Checkbutton(
        panel_izq,
        text="Quitar el fondo automáticamente (aislar objeto)",
        variable=var_quitar_fondo,
        command=lambda: actualizar_vista_previa(),
    )
    chk_fondo.pack(anchor="w", pady=4)

    chk_espejo = ttk.Checkbutton(
        panel_izq,
        text="Relleno automático trasero (duplicar en espejo 3D)",
        variable=var_relleno_espejo,
    )
    chk_espejo.pack(anchor="w", pady=4)

    chk_visor = ttk.Checkbutton(
        panel_izq,
        text="Abrir visor 3D interactivo (Open3D) al terminar",
        variable=var_abrir_visor,
    )
    chk_visor.pack(anchor="w", pady=4)

    ttk.Separator(panel_izq, orient="horizontal").pack(fill="x", pady=12)

    ttk.Label(panel_izq, text="3. Parámetros de Reconstrucción", style="Section.TLabel").pack(anchor="w", pady=(0, 8))

    grid_params = ttk.Frame(panel_izq, style="Card.TFrame")
    grid_params.pack(fill="x")
    grid_params.columnconfigure(1, weight=1)

    ttk.Label(grid_params, text="Modo de profundidad:").grid(row=0, column=0, sticky="w", pady=4, padx=(0, 8))
    combo_modo = ttk.Combobox(
        grid_params,
        textvariable=var_modo,
        values=["auto", "geometrico", "neural"],
        state="readonly",
        width=15,
    )
    combo_modo.grid(row=0, column=1, sticky="ew", pady=4)

    ttk.Label(grid_params, text="Clusters K-Means (K):").grid(row=1, column=0, sticky="w", pady=4, padx=(0, 8))
    spin_k = ttk.Spinbox(grid_params, from_=3, to=20, textvariable=var_k, width=10)
    spin_k.grid(row=1, column=1, sticky="ew", pady=4)

    ttk.Label(grid_params, text="Campo de visión (FOV°):").grid(row=2, column=0, sticky="w", pady=4, padx=(0, 8))
    spin_fov = ttk.Spinbox(grid_params, from_=35.0, to=100.0, increment=5.0, textvariable=var_fov, width=10)
    spin_fov.grid(row=2, column=1, sticky="ew", pady=4)

    ttk.Label(grid_params, text="Resolución máx. (px):").grid(row=3, column=0, sticky="w", pady=4, padx=(0, 8))
    spin_dim = ttk.Spinbox(grid_params, from_=320, to=1024, increment=80, textvariable=var_max_dim, width=10)
    spin_dim.grid(row=3, column=1, sticky="ew", pady=4)

    ttk.Label(grid_params, text="Carpeta de salida:").grid(row=4, column=0, sticky="w", pady=4, padx=(0, 8))
    entry_sal = ttk.Entry(grid_params, textvariable=var_salida, font=("Segoe UI", 9))
    entry_sal.grid(row=4, column=1, sticky="ew", pady=4)

    ttk.Separator(panel_izq, orient="horizontal").pack(fill="x", pady=14)

    btn_generar = ttk.Button(panel_izq, text="Generar Modelo 3D", style="Primary.TButton")
    btn_generar.pack(fill="x", pady=(4, 8))

    barra_progreso = ttk.Progressbar(panel_izq, orient="horizontal", mode="determinate", maximum=100)
    barra_progreso.pack(fill="x", pady=(0, 6))

    lbl_estado = ttk.Label(
        panel_izq,
        textvariable=var_estado,
        wraplength=340,
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
        text="Vista Previa (Original vs. Recorte de Fondo)",
        style="Section.TLabel",
    )
    lbl_titulo_previa.pack(side="left")
    ttk.Button(
        header_der,
        text="Actualizar vista previa",
        style="Secondary.TButton",
        command=lambda: actualizar_vista_previa(),
    ).pack(side="right")

    canvas_previa = tk.Label(panel_der, bg="#121318", fg=FG_SUB, text="Sin imagen seleccionada", font=("Segoe UI", 11))
    canvas_previa.grid(row=1, column=0, sticky="nsew")

    estado_preview = {"photo": None}

    def mostrar_array_en_canvas(arr_rgb: np.ndarray, titulo: str | None = None):
        if titulo:
            lbl_titulo_previa.config(text=titulo)
        cw = max(canvas_previa.winfo_width(), 520)
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
            if var_quitar_fondo.get() or mascara_alfa is not None:
                fg = segmentar_objeto_sin_fondo(img_bgr, mascara_alfa)
                # Crear fondo damero oscuro para apreciar el recorte
                h, w = img_rgb.shape[:2]
                yy, xx = np.mgrid[0:h, 0:w]
                damero = (((yy // 16) + (xx // 16)) % 2 == 0)
                bg_chess = np.where(damero[:, :, None], 56, 36).astype(np.uint8)
                recorte = np.where(fg[:, :, None], img_rgb, bg_chess)
                combinada = np.hstack([img_rgb, recorte])
                mostrar_array_en_canvas(
                    combinada,
                    "Vista Previa: Imagen Original (Izq.) | Objeto sin Fondo (Der.)",
                )
            else:
                mostrar_array_en_canvas(img_rgb, "Vista Previa: Imagen Original (Con fondo)")
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

        def tarea_fondo():
            try:
                res = ejecutar_pipeline(
                    ruta_imagen=ruta,
                    carpeta_salida=var_salida.get().strip() or "data/salida",
                    modo=var_modo.get(),
                    quitar_fondo=var_quitar_fondo.get(),
                    relleno_espejo=var_relleno_espejo.get(),
                    fov=float(var_fov.get()),
                    k=int(var_k.get()),
                    max_dim=int(var_max_dim.get()),
                    mostrar_visor=False,
                    callback_progreso=callback_ui,
                )

                def al_terminar():
                    btn_generar.config(state="normal")
                    # Mostrar vistas 3D generadas (incluyendo parte trasera 180°) en el panel
                    if os.path.exists(res["ruta_vistas"]):
                        vistas_bgr = cv2.imread(res["ruta_vistas"])
                        if vistas_bgr is not None:
                            vistas_rgb = cv2.cvtColor(vistas_bgr, cv2.COLOR_BGR2RGB)
                            mostrar_array_en_canvas(
                                vistas_rgb,
                                "Resultado 3D: Vistas Multi-ángulo (Incluye Vista Trasera en Espejo)",
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
    ap = argparse.ArgumentParser(description="Generador de modelos 3D a partir de una imagen.")
    ap.add_argument("--imagen", default="data/prueba.jpg", help="Ruta de la foto (PNG, JPG, WEBP, BMP, etc.)")
    ap.add_argument("--salida", default="data/salida", help="Carpeta donde se exportará el modelo 3D")
    ap.add_argument("--modo", choices=["auto", "neural", "geometrico"], default="auto")
    ap.add_argument("--fov", type=float, default=60.0, help="Campo de visión horizontal (grados)")
    ap.add_argument("--k", type=int, default=8, help="Número de clusters K-Means")
    ap.add_argument("--z-lejos", type=float, default=8.0, help="Profundidad del infinito relativa a lo más cercano (=1)")
    ap.add_argument("--umbral", type=float, default=0.20, help="Corte de malla en saltos de profundidad")
    ap.add_argument("--max-dim", type=int, default=640, help="Dimensión máxima de procesamiento en píxeles")
    ap.add_argument("--quitar-fondo", action="store_true", default=True, help="Quitar el fondo de la foto (activo por defecto)")
    ap.add_argument("--conservar-fondo", action="store_true", help="Conservar el fondo original de la escena")
    ap.add_argument("--relleno-espejo", action="store_true", default=True, help="Duplicar el modelo por detrás en espejo (activo por defecto)")
    ap.add_argument("--sin-espejo", action="store_true", help="Generar solo la superficie frontal sin relleno trasero")
    ap.add_argument("--sin-visor", action="store_true", help="No abrir la ventana de Open3D al finalizar")
    ap.add_argument("--sin-menu", action="store_true", help="Ejecutar directamente por consola sin abrir el menú principal")
    args = ap.parse_args()

    # Si no se pide --sin-menu, abrimos el Menú Principal interactivo
    if not args.sin_menu:
        abrir_menu_principal(args)
        return

    quitar_bg = not args.conservar_fondo
    espejo = not args.sin_espejo

    ejecutar_pipeline(
        ruta_imagen=args.imagen,
        carpeta_salida=args.salida,
        modo=args.modo,
        quitar_fondo=quitar_bg,
        relleno_espejo=espejo,
        fov=args.fov,
        k=args.k,
        z_lejos=args.z_lejos,
        umbral=args.umbral,
        max_dim=args.max_dim,
        mostrar_visor=not args.sin_visor,
    )


if __name__ == "__main__":
    main()
