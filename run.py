#!/usr/bin/env python3
"""
Lanzador unico de la aplicacion web de estabilizacion.

Uso:
    python run.py                 # arranca en http://127.0.0.1:8000
    python run.py --port 8080     # puerto personalizado
    python run.py --host 0.0.0.0  # accesible en la red local

Si el modelo de aprendizaje automatico aun no existe, se entrena automaticamente
la primera vez.
"""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)


def ensure_ml_model():
    model_path = os.path.join(ROOT_DIR, 'ml', 'adaptive_weights_model.pkl')
    if os.path.exists(model_path):
        return
    print('Modelo de pesos adaptativos no encontrado. Entrenando (una sola vez)...')
    from ml import train_adaptive_weights
    train_adaptive_weights.main()


def main():
    parser = argparse.ArgumentParser(description='Estabilizador de video con IA')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()

    ensure_ml_model()

    import uvicorn
    print(f'\n  Estabilizador de Video ML')
    print(f'  Abre tu navegador en:  http://{args.host}:{args.port}\n')
    uvicorn.run('backend.app:app', host=args.host, port=args.port, reload=args.reload)


if __name__ == '__main__':
    main()
