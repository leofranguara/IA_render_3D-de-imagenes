"""
Fase 0 (nueva): estimación de profundidad monocular con una red neuronal (MiDaS).

Por qué existe: las reglas heurísticas (horizonte + color + textura) no pueden saber
qué tan lejos está un objeto cuando la escena no es "una carretera con árboles".
Una red entrenada con millones de imágenes sí aprende pistas de perspectiva, tamaño
relativo, oclusión y desenfoque. Su salida alimenta a las fases 2 y 3, que siguen
usando superpíxeles y K-Means para regularizar y estructurar el resultado.

La red devuelve DISPARIDAD relativa (inverso de la profundidad, escala desconocida).
Aquí se normaliza a [0, 1] con 1 = cerca y 0 = lejos (misma convención que tu mapa Z).
"""
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # el resto del proyecto sigue funcionando en modo geométrico
    ort = None

URL_MIDAS_SMALL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELO_DEFECTO = RAIZ / "modelos" / "midas_small.onnx"

_MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def asegurar_modelo(ruta: Path = RUTA_MODELO_DEFECTO, url: str = URL_MIDAS_SMALL) -> Path:
    """Descarga el modelo (~64 MB) la primera vez. Devuelve la ruta local."""
    ruta = Path(ruta)
    if ruta.exists() and ruta.stat().st_size > 10_000_000:
        return ruta
    ruta.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Fase 0] Descargando modelo de profundidad a {ruta} ...")
    tmp = ruta.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, ruta)
    return ruta


class EstimadorProfundidadNeural:
    def __init__(self, ruta_modelo: Path | str | None = None):
        if ort is None:
            raise RuntimeError("Falta onnxruntime: pip install onnxruntime")
        ruta = asegurar_modelo(Path(ruta_modelo) if ruta_modelo else RUTA_MODELO_DEFECTO)
        self.sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
        entrada = self.sesion.get_inputs()[0]
        self.nombre_entrada = entrada.name
        forma = entrada.shape  # [1, 3, H, W]
        self.alto = forma[2] if isinstance(forma[2], int) else 256
        self.ancho = forma[3] if isinstance(forma[3], int) else 256

    def _inferir(self, img_rgb: np.ndarray) -> np.ndarray:
        x = cv2.resize(img_rgb, (self.ancho, self.alto), interpolation=cv2.INTER_CUBIC)
        x = (x.astype(np.float32) / 255.0 - _MEDIA) / _STD
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        y = self.sesion.run(None, {self.nombre_entrada: x})[0]
        return np.squeeze(y).astype(np.float32)

    def estimar(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        img_rgb: uint8 (H, W, 3). Devuelve disparidad float32 (H, W) en [0, 1], 1 = cerca.
        Se promedia con la imagen volteada (test-time augmentation) para reducir ruido.
        """
        H, W = img_rgb.shape[:2]
        d1 = self._inferir(img_rgb)
        d2 = self._inferir(np.ascontiguousarray(img_rgb[:, ::-1]))[:, ::-1]
        d = 0.5 * (d1 + d2)
        d = cv2.resize(np.ascontiguousarray(d), (W, H), interpolation=cv2.INTER_CUBIC)
        lo, hi = np.percentile(d, [2, 98])
        return np.clip((d - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)
