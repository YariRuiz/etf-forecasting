

import sys 
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_etf_data
from preprocessor import normalize, create_windows
from lstm_model import LSTMModel


def train_pipeline(
    filepath:str,
    symbol: str,
    window_size: int = 60,
    epochs: int = 50
): 
    """
    Pipeline completo de entrenamiento.

    Args: 
        filepath: ruta al CSV
        symbol: Simbolo del ETF
        window_size: Tamaño de la ventana Temporal
        epochs: Numero maximo de epocas

    Returns: 
        Tupla con (Modelo entrenado, min_val, max_val)
    """

    # Carga de datos
    print(f"Cargando datos de {symbol}...")
    prices = load_etf_data(filepath, symbol)

    # Normalizar 
    print("Normalizando...")
    normalized, min_val, max_val = normalize(prices)

    # Crear ventanas temporales
    print("Creando ventanas temporales...")
    X,y = create_windows(normalized, window_size)

    # Reshape LSTM (samples, timesteps, features)
    X = X.reshape((X.shape[0],X.shape[1], 1))

    # Entrenar
    print("Entenando Modelo...")
    model = LSTMModel(window_size=window_size)
    model.train(X,y, epochs=epochs)

    print("Entrenamiento Completado.")
    return model, min_val, max_val

if __name__ == "__main__":
    model, min_val, max_val = train_pipeline(
        filepath="data/etf-prices.csv",
        symbol="SPY",
        window_size=60,
        epochs=50
    )
    print(f"\nMin: {min_val:.2f} | Max: {max_val:.2f}")