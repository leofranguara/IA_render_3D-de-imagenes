"""
Fase 0: Estimación de profundidad monocular 100% OFFLINE (sin conexión a internet).

Si existe un archivo de modelo ONNX local en `modelos/midas_small.onnx` e `onnxruntime`
está instalado, puede usarse opcionalmente. De lo contrario, el sistema funciona
al 100% sin internet mediante el motor de inferencia geométrica, fotométrica y
estructural multi-escala de la Fase 2.
"""
from pathlib import Path
import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELO_DEFECTO = RAIZ / "modelos" / "midas_small.onnx"

_MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class EstimadorProfundidadNeural:
    def __init__(self, ruta_modelo: Path | str | None = None):
        if ort is None:
            raise RuntimeError("Modo 100% offline activo (onnxruntime no requerido).")
        ruta = Path(ruta_modelo) if ruta_modelo else RUTA_MODELO_DEFECTO
        if not ruta.exists() or ruta.stat().st_size < 1_000_000:
            raise FileNotFoundError(
                f"No se encontró modelo local en {ruta}. Se usará el motor 3D universal offline."
            )
        self.sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
        entrada = self.sesion.get_inputs()[0]
        self.nombre_entrada = entrada.name
        forma = entrada.shape
        self.alto = forma[2] if isinstance(forma[2], int) else 256
        self.ancho = forma[3] if isinstance(forma[3], int) else 256

    def _inferir(self, img_rgb: np.ndarray) -> np.ndarray:
        x = cv2.resize(img_rgb, (self.ancho, self.alto), interpolation=cv2.INTER_CUBIC)
        x = (x.astype(np.float32) / 255.0 - _MEDIA) / _STD
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        y = self.sesion.run(None, {self.nombre_entrada: x})[0]
        return np.squeeze(y).astype(np.float32)

    def estimar(self, img_rgb: np.ndarray) -> np.ndarray:
        H, W = img_rgb.shape[:2]
        d1 = self._inferir(img_rgb)
        d2 = self._inferir(np.ascontiguousarray(img_rgb[:, ::-1]))[:, ::-1]
        d = 0.5 * (d1 + d2)
        d = cv2.resize(np.ascontiguousarray(d), (W, H), interpolation=cv2.INTER_CUBIC)
        lo, hi = np.percentile(d, [2, 98])
        return np.clip((d - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)
