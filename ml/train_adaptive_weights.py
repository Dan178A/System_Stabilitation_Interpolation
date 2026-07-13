"""
Entrenamiento del modelo de aprendizaje automatico que predice los pesos
adaptativos (lambda_t) de la funcion de energia del estabilizador.

En la version original del sistema, los pesos adaptativos se calculaban con el
modelo LINEAL fijo del paper (coeficientes -1.93, 0.95, 5.83, 4.88). La tesis
promete que estos pesos sean PREDICHOS por un modelo de aprendizaje automatico;
este script implementa exactamente eso: entrena una red neuronal (perceptron
multicapa, MLPRegressor) que aprende a mapear descriptores de movimiento a un
peso adaptativo, generalizando de forma suave y no lineal.

Entradas del modelo (features por fotograma):
    1. translational_element  -> magnitud de traslacion normalizada.
    2. affine_component        -> razon de valores propios de la parte afin.
    3. translational * affine  -> termino de interaccion.
    4. translational^2         -> termino cuadratico.

Salida:
    lambda_t (peso adaptativo, no negativo).

DATOS DE ENTRENAMIENTO
----------------------
Se generan de forma sintetica muestras de descriptores de movimiento que siguen
la estadistica del temblor de mano en video movil (traslaciones pequenas y
componentes afines cercanas a 1). El objetivo (target) se construye a partir de
un "maestro" que codifica el conocimiento del dominio:

    - Mas deformacion afin rigida (affine_component -> 1) => es seguro suavizar
      con fuerza (lambda alto): el movimiento es un temblor a corregir.
    - Mas traslacion intencional (paneos) => conviene suavizar menos (lambda
      bajo) para no recortar en exceso.

El maestro parte del modelo lineal original y lo ajusta hacia una preferencia de
estabilidad, y el MLP aprende una version suave y generalizable de esa relacion.
El resultado NO es una copia del modelo lineal: es una funcion no lineal
aprendida, reentrenable sobre features reales extraidas de videos etiquetados.

Uso:
    python ml/train_adaptive_weights.py
"""

import os
import pickle

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


RANDOM_SEED = 42
MODEL_FILENAME = 'adaptive_weights_model.pkl'


def generate_training_data(num_samples=20000, seed=RANDOM_SEED):
    """Genera descriptores de movimiento sinteticos y sus pesos objetivo."""
    rng = np.random.default_rng(seed)

    # Traslacion normalizada: temblor de mano tipico, mayoria de valores pequenos.
    translational = np.abs(rng.gamma(shape=2.0, scale=0.03, size=num_samples))
    translational = np.clip(translational, 0.0, 0.4)

    # Componente afin: cercana a 1 (poca deformacion), con cola hacia valores menores.
    affine = 1.0 - np.abs(rng.gamma(shape=2.0, scale=0.04, size=num_samples))
    affine = np.clip(affine, 0.75, 1.0)

    # --- Maestro: conocimiento de dominio para el peso objetivo ---
    # Base del modelo lineal original.
    candidate_translation = -1.93 * translational + 0.95
    candidate_affine = 5.83 * affine + 4.88
    linear_teacher = np.maximum(np.minimum(candidate_translation, candidate_affine), 0.0)

    # Ajuste orientado a estabilidad: premiar movimiento rigido (affine alto) y
    # penalizar traslaciones grandes (paneos intencionales) de forma no lineal.
    stability_bonus = 3.0 * (affine - 0.9)            # +/- segun rigidez
    pan_penalty = 6.0 * translational ** 1.5          # penaliza paneos fuertes
    target = linear_teacher + stability_bonus - pan_penalty

    # Ruido leve para robustez y para evitar memorizacion exacta del maestro.
    target += rng.normal(0.0, 0.15, size=num_samples)
    target = np.clip(target, 0.0, 12.0)

    features = np.column_stack([
        translational,
        affine,
        translational * affine,
        translational ** 2,
    ])
    return features, target


def build_model():
    """Perceptron multicapa con normalizacion de entradas."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            alpha=1e-3,
            max_iter=800,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=RANDOM_SEED,
        )),
    ])


def main():
    print('Generando datos de entrenamiento sinteticos...')
    features, target = generate_training_data()

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_SEED
    )

    print(f'Entrenando MLPRegressor sobre {len(x_train)} muestras...')
    model = build_model()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))
    print(f'Evaluacion en test -> RMSE: {rmse:.4f}  |  R2: {r2:.4f}')

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
    with open(output_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f'Modelo guardado en <{output_path}>')

    # Prueba rapida de cordura: casos extremos.
    demo = np.array([
        [0.01, 0.99, 0.01 * 0.99, 0.01 ** 2],   # casi estatico + rigido -> lambda alto
        [0.30, 0.90, 0.30 * 0.90, 0.30 ** 2],   # paneo fuerte           -> lambda bajo
    ])
    demo_pred = np.maximum(model.predict(demo), 0.0)
    print(f'Cordura -> estatico/rigido: {demo_pred[0]:.2f}  |  paneo fuerte: {demo_pred[1]:.2f}')
    return output_path


if __name__ == '__main__':
    main()
