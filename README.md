# Reconstrucción 3D Monocular mediante Aprendizaje No Supervisado (SLIC y K-Means)

Generador universal de modelos 3D (`.obj`, `.ply`, `.glb`) a partir de una sola fotografía, basado **100 % en Aprendizaje No Supervisado y Visión por Computadora** (SLIC + K-Means + inferencia geométrica de profundidad invariante al albedo). Funciona **completamente offline en CPU**, sin redes neuronales preentrenadas ni descargas externas.

---

## Requisitos
- **Python 3.10+** (probado en Python 3.12 / 3.13 en Windows).
- Dependencias listadas en `requirements.txt` (`numpy`, `scipy`, `opencv-python`, `pillow`, `scikit-image`, `scikit-learn`, `matplotlib`, `trimesh`, y `open3d` opcional para el visor 3D interactivo).

---

## Instalación

Desde la carpeta raíz del proyecto en **PowerShell**:

```powershell
# 1. Crear entorno virtual (si aún no existe)
python -m venv venv

# 2. Activar el entorno virtual
.\venv\Scripts\Activate.ps1

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Cómo ejecutar el proyecto

### Opción 1: Lanzador rápido en Windows (`ejecutar.bat`)
- Haz **doble clic** en `ejecutar.bat` para abrir directamente el **Menú Principal (GUI)** usando el entorno virtual `venv`.
- También puedes pasar argumentos desde consola:
  ```bat
  ejecutar.bat --imagen data\prueba.jpg --sin-menu
  ```

### Opción 2: Interfaz gráfica (Tkinter) desde consola
```powershell
.\venv\Scripts\Activate.ps1
python src/main.py
```
En la interfaz gráfica puedes:
- Seleccionar cualquier imagen (`.jpg`, `.png`, `.webp`, `.bmp`, `.tif`, etc.).
- Elegir el manejo del fondo (`Automático`, `Quitar` o `Conservar`).
- Activar el relleno trasero en espejo y ajustar su grosor (`15%` a `140%`).
- Configurar `K` (clústeres K-Means), `FOV°` y resolución máxima.
- Ver en vivo cada etapa (**`SLIC`** → **`Secciones K-Means`** → **`Vistas 3D`**) y alternar entre **`SLIC`**, **`Secciones K-Means`**, **`Diagnóstico`**, **`Vistas 3D`** y **`Vista previa`**.

### Opción 3: Ejecución directa por consola (CLI)
```powershell
# Con visor interactivo Open3D al terminar
python src/main.py --imagen data/prueba.jpg --sin-menu

# Sin GUI ni visor (modo rápido / por lotes)
python src/main.py --imagen data/prueba.jpg --sin-menu --sin-visor
```

---

## Parámetros útiles por consola

| Parámetro | Por defecto | Descripción |
| :--- | :--- | :--- |
| `--imagen <ruta>` | `data/prueba.jpg` | Ruta de la fotografía de entrada. |
| `--salida <carpeta>` | `data/salida` | Carpeta donde se guardan los archivos generados. |
| `--fondo <modo>` | `auto` | `auto` (detecta según escena), `quitar` (aísla objeto) o `conservar` (escena completa). |
| `--k <int>` | `8` | Número de clústeres de K-Means (`3`–`20`). |
| `--fov <float>` | `60.0` | Campo de visión horizontal en grados. |
| `--max-dim <int>` | `640` | Dimensión máxima de procesamiento en píxeles. |
| `--grosor-espejo <float>` | `0.65` | Factor de grosor en modo espejo (`0.25` muy delgado a `1.0` normal). |
| `--sin-espejo` | *(inactivo)* | Genera solo la superficie frontal sin cierre trasero. |
| `--sin-visor` | *(inactivo)* | No abre la ventana interactiva de Open3D al terminar. |
| `--sin-menu` | *(inactivo)* | Ejecuta directamente por consola sin abrir la interfaz Tkinter. |

---

## Controles del visor 3D (Open3D)
- **Clic izquierdo + arrastrar:** rotar el modelo en 360°.
- **Clic central (rueda) + arrastrar** o **`Ctrl` + Clic izquierdo + arrastrar:** desplazarse (pan) en el espacio 3D.
- **Rueda del mouse:** acercar / alejar (zoom).

---

## Archivos generados
Por defecto, en `data/salida/` se exportan:
- `modelo_3d.obj`, `modelo_3d.ply`, `modelo_3d.glb`: malla tridimensional exportada.
- `slic_superpixeles.png`: malla de superpíxeles SLIC y mosaico de color promedio.
- `kmeans_secciones.png`: división por secciones de K-Means (`C0..CK-1`).
- `diagnostico.png`: resumen comparativo de 4 etapas.
- `profundidad.png`: mapa de profundidad final en escala de grises.
- `vistas.png`: renderizado multi-ángulo (`-45°`, `-20°`, `0°`, `+20°`, `+45°`, `135°`, `180°`).
- `sin_fondo.png`: recorte RGBA del objeto (cuando se aplica segmentación de primer plano).

---

## Estructura del proyecto
```text
IA_render_3D-de-imagenes/
├── ejecutar.bat                      # Lanzador rápido en Windows
├── requirements.txt                  # Dependencias de Python
├── data/                             # Imágenes de prueba y carpeta de salida
└── src/
    ├── main.py                       # Orquestador, CLI y GUI (Tkinter)
    ├── fase1_slic.py                 # Carga, clasificación, segmentación y SLIC
    ├── fase2_clustering_profundidad.py # K-Means e inferencia de profundidad 3D
    └── fase3_malla_3d.py             # Malla 3D, cierre espejo, vistas y visor Open3D
```
