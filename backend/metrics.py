"""
Metricas de calidad de imagen para evaluar la estabilizacion.

Se implementan MSE, RMSE, PSNR y SSIM (las mismas metricas que usa la tesis) con
NumPy y OpenCV, sin dependencias pesadas. Para medir ESTABILIDAD se comparan
fotogramas consecutivos: un video mas estable tiene menor diferencia entre
fotogramas contiguos (menor MSE, mayor PSNR y SSIM inter-fotograma).
"""

import cv2
import numpy as np


def _to_gray(frame):
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def mse(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(a, b):
    error = mse(a, b)
    if error == 0:
        return 100.0
    return float(20.0 * np.log10(255.0) - 10.0 * np.log10(error))


def ssim(a, b):
    """SSIM global (una sola ventana) sobre imagenes en escala de grises."""
    a = _to_gray(a).astype(np.float64)
    b = _to_gray(b).astype(np.float64)
    k1, k2, data_range = 0.01, 0.03, 255.0
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(numerator / denominator)


def _read_frames(video_path, max_frames=60, scale=0.5):
    """Lee hasta max_frames fotogramas (submuestreados) en escala reducida."""
    capture = cv2.VideoCapture(video_path)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    step = max(1, total // max_frames)
    frames = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            if scale != 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(_to_gray(frame))
        index += 1
    capture.release()
    return frames


def interframe_stability(video_path, max_frames=60):
    """
    Promedio de MSE, PSNR y SSIM entre fotogramas consecutivos. Menor MSE y mayor
    PSNR/SSIM indican mayor estabilidad temporal.
    """
    frames = _read_frames(video_path, max_frames=max_frames)
    if len(frames) < 2:
        return {'mse': 0.0, 'rmse': 0.0, 'psnr': 100.0, 'ssim': 1.0}

    # Alinear tamanos por si el recorte cambio dimensiones.
    height = min(f.shape[0] for f in frames)
    width = min(f.shape[1] for f in frames)
    frames = [f[:height, :width] for f in frames]

    mses, psnrs, ssims = [], [], []
    for i in range(len(frames) - 1):
        mses.append(mse(frames[i], frames[i + 1]))
        psnrs.append(psnr(frames[i], frames[i + 1]))
        ssims.append(ssim(frames[i], frames[i + 1]))

    mean_mse = float(np.mean(mses))
    return {
        'mse': round(mean_mse, 3),
        'rmse': round(float(np.sqrt(mean_mse)), 3),
        'psnr': round(float(np.mean(psnrs)), 3),
        'ssim': round(float(np.mean(ssims)), 4),
    }


def compare_videos(original_path, stabilized_path, max_frames=60):
    """
    Devuelve metricas de estabilidad temporal del video original y del estabilizado,
    junto con la mejora relativa (positiva = mejor).
    """
    original = interframe_stability(original_path, max_frames)
    stabilized = interframe_stability(stabilized_path, max_frames)

    def improvement(before, after, higher_is_better):
        if before == 0:
            return 0.0
        delta = (after - before) / abs(before) * 100.0
        return round(delta if higher_is_better else -delta, 1)

    return {
        'original': original,
        'stabilized': stabilized,
        'improvement': {
            'mse': improvement(original['mse'], stabilized['mse'], higher_is_better=False),
            'psnr': improvement(original['psnr'], stabilized['psnr'], higher_is_better=True),
            'ssim': improvement(original['ssim'], stabilized['ssim'], higher_is_better=True),
        },
    }
