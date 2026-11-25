# AGENTS.md

Guía de contexto, arquitectura y reglas de trabajo para agentes que colaboren en **IA_render_3D-de-imagenes**.

## 1. Descripción general del proyecto
Generador universal de modelos 3D (`.obj`) a partir de una sola fotografía (objetos, vehículos, retratos, casas o paisajes), diseñado para funcionar **100% offline** con soporte tanto por **interfaz gráfica (Tkinter)** como por **consola (CLI)**.

## 2. Estructura del repositorio
- `ejecutar.bat`: lanzador en Windows que usa `venv\Scripts\python.exe` (o `python` del sistema como respaldo) para ejecutar `src\main.py` reenviando argumentos.
- `requirements.txt`: dependencias del proyecto (`numpy`, `scipy`, `opencv-python`, `pillow`, `scikit-image`, `scikit-learn`, `matplotlib`, `trimesh`, `open3d` opcional para el visor interactivo).
- `data/`: imágenes de entrada (ej. `data/prueba.jpg`) y carpeta de salida por defecto (`data/salida/`).
- `src/`:
  - `src/main.py`: punto de entrada (`main`), orquestación del pipeline (`ejecutar_pipeline`), diagnóstico visual (`guardar_diagnostico`) y GUI (`abrir_menu_principal`).
  - `src/fase1_slic.py`: carga multi-formato (`cargar_imagen_cualquier_formato`), estimación de horizonte/punto de fuga (`estimar_punto_fuga_hough`), clasificación de escena (`clasificar_tipo_escena`), segmentación de primer plano sin fondo (`segmentar_objeto_sin_fondo`) y superpíxeles SLIC (`procesar_superpixeles`).
  - `src/fase2_clustering_profundidad.py`: agrupación K-Means sobre superpíxeles y motor universal de inferencia de profundidad invariante al albedo (`agrupar_e_inferir_profundidad`), con modelos diferenciados para retratos, objetos/vehículos (incluyendo perfil cilíndrico y detección de cavidades superiores) y escenas completas.
  - `src/fase3_malla_3d.py`: construcción de malla 3D (`construir_malla_volumetrica`) para objetos aislados o escenas completas, soporte de cavidades interiores con respaldo convexo, cierre trasero simétrico en modo espejo (`relleno_espejo`, `grosor_espejo`), renderizado de vistas (`guardar_vistas`) y exportación/visor (`visualizar_y_exportar`).

## 3. Cómo ejecutar el proyecto
Desde la raíz del repositorio (en Windows / PowerShell):

```powershell
# Activar entorno virtual
.\venv\Scripts\Activate.ps1

# Abrir Menú Principal (Tkinter)
python src/main.py

# Ejecución directa por consola (sin GUI ni visor Open3D)
python src/main.py --imagen data/prueba.jpg --sin-menu --sin-visor
```

También se puede iniciar directamente con `ejecutar.bat` (doble clic para abrir la GUI, o pasando argumentos desde consola).

## 4. Reglas y prioridades obligatorias
1. **100% Offline (sin descargas ni llamadas de red):**
   - Nunca agregues descargas automáticas de pesos, modelos ni llamadas a APIs externas en tiempo de ejecución.
   - Cualquier modelo neural (como `modelos/midas_small.onnx` en `fase0_profundidad_neural.py`) debe ser estrictamente opcional y local, manteniendo el motor geométrico/fotométrico de `fase2` completamente funcional por defecto.
2. **Paridad entre GUI y CLI:**
   - Todo parámetro o funcionalidad nueva expuesta en `ejecutar_pipeline` o en el CLI (`argparse` en `main()`) debe reflejarse en `abrir_menu_principal()` (Tkinter) y viceversa, manteniendo consistentes los valores por defecto.
3. **Calidad de malla 3D, segmentación y rendimiento:**
   - Preserva la invarianza al albedo en `fase2_clustering_profundidad.py` para evitar que zonas oscuras (cabello, llantas, vidrios) se hundan o zonas claras se inflen.
   - Cuida la geometría de bordes y el cierre en modo espejo en `fase3_malla_3d.py`, evitando islas flotantes y distorsiones extremas.
   - Prefiere operaciones vectorizadas con NumPy/OpenCV/SciPy para mantener tiempos de ejecución bajos sobre imágenes redimensionadas (`max_dim=640` por defecto).
4. **Idioma y estilo del código:**
   - Redacta comentarios, docstrings, mensajes de consola y textos de la interfaz en **español**, manteniendo la coherencia con el resto del proyecto.
   - Conserva los docstrings y comentarios existentes que no estén relacionados con tu cambio.

## 5. Verificación requerida antes de finalizar cambios
Tras modificar código en `src/`, ejecuta una prueba rápida de extremo a extremo por consola sin abrir ventanas interactivas:

```powershell
.\venv\Scripts\python.exe src/main.py --imagen data/prueba.jpg --sin-menu --sin-visor
```

Verifica que:
- El comando termine con código de salida `0` y sin excepciones.
- Se generen correctamente los archivos esperados en `data/salida/`:
  - `diagnostico.png`
  - `profundidad.png`
  - `vistas.png`
  - `modelo_3d.obj`
  - `sin_fondo.png` (cuando se aplica segmentación de primer plano).
- Si modificaste la interfaz gráfica (`abrir_menu_principal`), valida también que `src/main.py` importe y arranque sin errores de Tkinter.
