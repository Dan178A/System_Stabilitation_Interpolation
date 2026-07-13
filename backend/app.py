"""
Backend FastAPI para el sistema de estabilizacion de video.

Expone el motor Stabilizer a traves de una API web:

    GET  /                      -> interfaz web (frontend/index.html)
    POST /api/stabilize         -> sube un video + parametros, arranca un trabajo
    GET  /api/progress/{job}    -> flujo SSE con el progreso en vivo
    GET  /api/result/{job}      -> metricas y rutas de los videos (JSON)
    GET  /api/video/{job}/{kind}-> sirve el video original o el estabilizado
    GET  /api/methods           -> catalogo de metodos de pesos adaptativos

La estabilizacion se ejecuta en un hilo de trabajo; el callback de progreso del
Stabilizer actualiza el estado del trabajo, que el endpoint SSE transmite al
navegador. El video se escribe con OpenCV (codec mp4v) y se transcodifica con
ffmpeg a H.264 para reproduccion universal en el navegador.
"""

import json
import os
import shutil
import subprocess
import threading
import time
import uuid

import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import (
    FileResponse, StreamingResponse, JSONResponse, HTMLResponse,
)

# Importaciones del proyecto (raiz del repo en sys.path via run.py).
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from stabilizer import Stabilizer
from backend import metrics as metrics_module

FRONTEND_DIR = os.path.join(ROOT_DIR, 'frontend')
JOBS_DIR = os.path.join(ROOT_DIR, 'backend', '_jobs')
os.makedirs(JOBS_DIR, exist_ok=True)

app = FastAPI(title='Estabilizador de Video ML', version='2.0')

# Registro de trabajos en memoria (id -> estado).
JOBS = {}
JOBS_LOCK = threading.Lock()

METHODS = {
    'ml': {
        'id': Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ML,
        'name': 'Aprendizaje automatico',
        'description': 'Pesos adaptativos predichos por una red neuronal entrenada.',
        'badge': 'IA',
    },
    'original': {
        'id': Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL,
        'name': 'Modelo lineal (paper)',
        'description': 'Modelo lineal clasico del paper original. Equilibrado.',
        'badge': 'Clasico',
    },
    'constant_high': {
        'id': Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH,
        'name': 'Estabilidad maxima',
        'description': 'Suavizado agresivo. Mayor estabilidad, mas recorte.',
        'badge': 'Fuerte',
    },
    'constant_low': {
        'id': Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW,
        'name': 'Recorte minimo',
        'description': 'Suavizado suave. Menos recorte, ligero temblor.',
        'badge': 'Suave',
    },
}


def _set_job(job_id, **kwargs):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {})
        JOBS[job_id].update(kwargs)


def _get_job(job_id):
    with JOBS_LOCK:
        return dict(JOBS.get(job_id, {}))


def _transcode_h264(source_path, target_path):
    """Transcodifica a H.264/yuv420p con ffmpeg para reproduccion en el navegador."""
    if not shutil.which('ffmpeg'):
        # Sin ffmpeg: usar el archivo tal cual (puede no reproducir en algunos navegadores).
        shutil.copy(source_path, target_path)
        return
    subprocess.run(
        ['ffmpeg', '-y', '-i', source_path,
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
         target_path],
        check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _run_stabilization(job_id, input_path, params):
    """Ejecuta la estabilizacion en un hilo de trabajo."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    raw_out = os.path.join(job_dir, 'stabilized_raw.mp4')
    web_out = os.path.join(job_dir, 'stabilized.mp4')
    web_original = os.path.join(job_dir, 'original.mp4')

    stage_labels = {
        'reading': 'Leyendo video',
        'motion': 'Estimando movimiento (malla)',
        'optimizing': 'Minimizando funcion de energia',
        'warping': 'Interpolando y deformando',
        'metrics': 'Calculando metricas',
        'writing': 'Escribiendo video',
        'done': 'Finalizado',
    }

    def progress_callback(stage, fraction):
        _set_job(job_id,
                 stage=stage,
                 stage_label=stage_labels.get(stage, stage),
                 progress=round(fraction, 3))

    try:
        _set_job(job_id, status='running', stage='reading',
                 stage_label='Leyendo video', progress=0.0)

        stabilizer = Stabilizer(
            mesh_row_count=params['mesh'],
            mesh_col_count=params['mesh'],
            temporal_smoothing_radius=params['smoothing'],
            optimization_num_iterations=params['iterations'],
            processing_scale=params['scale'],
            max_frames=params['max_frames'],
            output_fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            progress_callback=progress_callback,
        )

        started = time.time()
        cropping_ratio, distortion_score, stability_score = stabilizer.stabilize(
            input_path, raw_out, adaptive_weights_definition=params['method_id'],
        )
        elapsed = round(time.time() - started, 2)

        _set_job(job_id, stage='writing', stage_label='Transcodificando (H.264)', progress=0.97)
        _transcode_h264(raw_out, web_out)
        _transcode_h264(input_path, web_original)

        quality = metrics_module.compare_videos(web_original, web_out)

        result = {
            'engine_metrics': {
                'cropping_ratio': round(float(cropping_ratio), 4),
                'distortion_score': round(float(distortion_score), 4),
                'stability_score': round(float(stability_score), 4),
            },
            'quality_metrics': quality,
            'elapsed_seconds': elapsed,
            'method': params['method_key'],
        }
        _set_job(job_id, status='done', stage='done', stage_label='Finalizado',
                 progress=1.0, result=result)
    except Exception as error:  # pragma: no cover - reporte de error al frontend
        import traceback
        traceback.print_exc()
        _set_job(job_id, status='error', error=str(error), progress=1.0)


@app.get('/', response_class=HTMLResponse)
def index():
    index_path = os.path.join(FRONTEND_DIR, 'index.html')
    if not os.path.exists(index_path):
        return HTMLResponse('<h1>Frontend no encontrado</h1>', status_code=404)
    with open(index_path, 'r', encoding='utf-8') as handle:
        return HTMLResponse(handle.read())


@app.get('/api/methods')
def get_methods():
    return JSONResponse({
        key: {k: v for k, v in value.items() if k != 'id'}
        for key, value in METHODS.items()
    })


@app.post('/api/stabilize')
async def stabilize_endpoint(
    video: UploadFile = File(...),
    method: str = Form('ml'),
    mesh: int = Form(16),
    smoothing: int = Form(10),
    iterations: int = Form(100),
    scale: float = Form(1.0),
    max_frames: int = Form(0),
):
    if method not in METHODS:
        raise HTTPException(status_code=400, detail=f'Metodo invalido: {method}')

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    input_path = os.path.join(job_dir, 'input_' + os.path.basename(video.filename or 'video.mp4'))
    with open(input_path, 'wb') as handle:
        shutil.copyfileobj(video.file, handle)

    params = {
        'method_key': method,
        'method_id': METHODS[method]['id'],
        'mesh': max(4, min(int(mesh), 32)),
        'smoothing': max(1, min(int(smoothing), 30)),
        'iterations': max(10, min(int(iterations), 300)),
        'scale': max(0.25, min(float(scale), 1.0)),
        'max_frames': int(max_frames) if int(max_frames) > 0 else None,
    }

    _set_job(job_id, status='queued', progress=0.0, filename=video.filename)
    thread = threading.Thread(
        target=_run_stabilization, args=(job_id, input_path, params), daemon=True,
    )
    thread.start()
    return JSONResponse({'job_id': job_id})


@app.get('/api/progress/{job_id}')
def progress_stream(job_id: str):
    def event_generator():
        last = None
        while True:
            job = _get_job(job_id)
            if not job:
                yield f'data: {json.dumps({"status": "unknown"})}\n\n'
                return
            payload = {
                'status': job.get('status'),
                'stage': job.get('stage'),
                'stage_label': job.get('stage_label'),
                'progress': job.get('progress', 0.0),
            }
            snapshot = json.dumps(payload)
            if snapshot != last:
                yield f'data: {snapshot}\n\n'
                last = snapshot
            if job.get('status') in ('done', 'error'):
                return
            time.sleep(0.4)

    return StreamingResponse(event_generator(), media_type='text/event-stream')


@app.get('/api/result/{job_id}')
def get_result(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Trabajo no encontrado')
    if job.get('status') == 'error':
        return JSONResponse({'status': 'error', 'error': job.get('error')}, status_code=500)
    if job.get('status') != 'done':
        return JSONResponse({'status': job.get('status'), 'progress': job.get('progress', 0.0)})
    return JSONResponse({'status': 'done', **job['result']})


@app.get('/api/video/{job_id}/{kind}')
def get_video(job_id: str, kind: str):
    if kind not in ('original', 'stabilized'):
        raise HTTPException(status_code=400, detail='kind invalido')
    path = os.path.join(JOBS_DIR, job_id, f'{kind}.mp4')
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='Video no encontrado')
    return FileResponse(path, media_type='video/mp4')
