# Estabilizador de Video · Interpolación + Aprendizaje Automático

Sistema de estabilización de video para dispositivos móviles que combina estimación de
movimiento por malla, interpolación y **aprendizaje automático**, con una **interfaz web
moderna** para subir un video, elegir el método, ver el progreso en vivo y comparar el
resultado antes/después con métricas de calidad.

> Trabajo de Investigación — Licenciatura en Computación, Universidad del Zulia.
> Autor: Daniel Alejandro Silva Rojas.

Esta versión moderniza el proyecto original (que era solo línea de comandos) con tres
aportes nuevos:

1. **Interfaz web funcional** (FastAPI + frontend). Subes un video, ajustas parámetros y
   obtienes una comparación interactiva antes/después con un dashboard de métricas
   (MSE, RMSE, PSNR, SSIM).
2. **Pesos adaptativos predichos por una red neuronal** — lo que la tesis proponía pero el
   código no implementaba. Un modelo `MLPRegressor` aprende a mapear descriptores de
   movimiento a los pesos de la función de energía.
3. **Optimización de rendimiento** — modo de procesado a resolución reducida, límite de
   fotogramas para previsualización, extracción de descriptores vectorizada y blindaje del
   suavizado temporal para clips cortos.

---

## Cómo se estabiliza un video (4 pasos)

1. **Estimación de movimiento:** se coloca una malla sobre el video y se rastrea cómo se
   mueve cada vértice usando detección y seguimiento de características (flujo óptico).
2. **Minimización de energía:** se calcula cómo debe moverse cada vértice en el video
   estabilizado minimizando una función de energía (método de Jacobi). Los **pesos
   adaptativos** de esa función controlan el equilibrio entre estabilidad y recorte.
3. **Interpolación y deformación:** se deforma cada fotograma para que los vértices sigan su
   trayectoria estabilizada, mapeando los píxeles con interpolación.
4. **Recorte y redimensionamiento:** se recortan los bordes inestables y se reescala al
   tamaño original.

---

## Inicio rápido (interfaz web)

```bash
# 1. Crear entorno e instalar dependencias
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

# 2. Arrancar la aplicación (entrena el modelo ML la primera vez)
python run.py
```

Abre el navegador en **http://127.0.0.1:8000**, arrastra un video y pulsa *Estabilizar*.

Para reproducción de video en el navegador se recomienda tener **ffmpeg** instalado (el
backend transcodifica la salida a H.264). Si no está, se sirve el archivo tal cual.

### Métodos de pesos adaptativos disponibles en la interfaz

| Método | Descripción |
|--------|-------------|
| **Aprendizaje automático (IA)** | Pesos predichos por la red neuronal entrenada. |
| Modelo lineal (paper) | Modelo lineal clásico del paper original. Equilibrado. |
| Estabilidad máxima | Suavizado agresivo: más estable, más recorte. |
| Recorte mínimo | Suavizado suave: menos recorte, ligero temblor. |

---

## El modelo de aprendizaje automático

El módulo `ml/` implementa la predicción de pesos adaptativos.

- `ml/train_adaptive_weights.py` entrena un **perceptrón multicapa** (`MLPRegressor`,
  scikit-learn) que aprende a mapear cuatro descriptores de movimiento por fotograma
  (traslación normalizada, componente afín y términos derivados) al peso adaptativo λ_t.
- El objetivo de entrenamiento parte del modelo lineal del paper como "maestro" y lo ajusta
  hacia una preferencia de estabilidad; el modelo aprende una versión suave, no lineal y
  generalizable, **reentrenable sobre descriptores reales** extraídos de videos etiquetados.
- El artefacto entrenado se guarda en `ml/adaptive_weights_model.pkl` y el motor lo carga de
  forma perezosa. Si no existe, el sistema recae de forma segura en el modelo lineal.

Reentrenar el modelo:

```bash
python ml/train_adaptive_weights.py
```

---

## Uso por código (API del motor)

El motor sigue disponible como biblioteca. Nuevos parámetros en negrita.

```python
import cv2
from stabilizer import Stabilizer

stabilizer = Stabilizer(
    mesh_row_count=16, mesh_col_count=16,
    temporal_smoothing_radius=10,
    processing_scale=0.5,          # NUEVO: procesa a resolución reducida (más rápido)
    max_frames=None,               # NUEVO: limita fotogramas (previsualización)
    output_fourcc=cv2.VideoWriter_fourcc(*'mp4v'),  # NUEVO: códec de salida
    progress_callback=lambda etapa, frac: print(etapa, frac),  # NUEVO: progreso
)

crop, distortion, stability = stabilizer.stabilize(
    'videos/video-1/video-1.m4v',
    'salida.mp4',
    adaptive_weights_definition=Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ML,  # NUEVO método
)
```

`stabilize` devuelve `(cropping_ratio, distortion_score, stability_score)` como en el
original.

---

## Métricas de calidad

La interfaz mide la **estabilidad temporal** comparando fotogramas consecutivos (un video
más estable tiene menor diferencia entre fotogramas contiguos):

- **MSE / RMSE** — error cuadrático medio (menor es mejor).
- **PSNR** — relación señal-ruido de pico en dB (mayor es mejor).
- **SSIM** — similitud estructural, 0 a 1 (cercano a 1 es mejor).

Se muestran los valores del video original y del estabilizado junto con la mejora relativa.
También se reportan las métricas del motor del paper: ratio de recorte, distorsión y
estabilidad.

---

## Estructura del proyecto

```
stabilizer.py                     Motor de estabilización (clase Stabilizer)
run.py                            Lanzador de la aplicación web
requirements.txt                  Dependencias
ml/
  train_adaptive_weights.py       Entrenamiento del modelo de pesos adaptativos
  adaptive_weights_model.pkl      Modelo entrenado (se genera al entrenar)
backend/
  app.py                          API FastAPI (upload, progreso SSE, resultados, video)
  metrics.py                      Cálculo de MSE / RMSE / PSNR / SSIM
frontend/
  index.html                      Interfaz web (subida, ajustes, comparación, métricas)
videos/                           Videos de demostración
```

## API HTTP

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Interfaz web |
| GET | `/api/methods` | Catálogo de métodos |
| POST | `/api/stabilize` | Sube video + parámetros, inicia un trabajo |
| GET | `/api/progress/{job}` | Progreso en vivo (SSE) |
| GET | `/api/result/{job}` | Métricas y estado (JSON) |
| GET | `/api/video/{job}/{original\|stabilized}` | Sirve los videos |

---

## Créditos

Videos de demostración en `videos/credits.txt`. Algoritmo base de estabilización por malla
con minimización de energía (MeshFlow); esta implementación fue construida para
experimentación y ampliada con aprendizaje automático e interfaz web.
