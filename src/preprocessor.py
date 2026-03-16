

import numpy as np

def normalize(data: np.ndarray) -> tuple[np.array , float, float]:
    """
    Normaliza los datos del rango [0,1]
    
    Args: 
        data: Array con los precios del ETF
        
    Retorna:
    Tupla con datos normalizados, valor minimo, valor maximo)
    """
    min_val = data.min()
    max_val = data.max()
    normalized = (data - min_val) / (max_val - min_val)
    return normalized, min_val, max_val


def denormalize(data: np.ndarray, min_val:float, max_val: float) -> np.ndarray:
    """
    Revierte la normalizacion a precios reales
    
    Args: 
        data: Array Normalizado
        min_val: Valor minimo original
        max_val: Valor maximo original
        
    Retorna:
        Array con precios en es cala original
    """
    return data * (max_val - min_val) + min_val

def create_windows(data: np.ndarray, window_size: int )-> tuple[np.ndarray, np.ndarray]:
    """
    Crea ventanas temporales para entrenar el LSTM
    
    Args:
        data: Array normalizado de precios
        window_size: Tamano de la ventana temporal 
        
    Retorna una tupla X, y donde X son las ventanas y y son los valores a predecir 
    
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size]) #Ventana de 60 dias
        y.append(data[i + window_size]) #Dia siguiente a predecir
    return np.array(X), np.array(y)


if __name__ == "__main__":
    from data_loader import load_etf_data
    
    prices = load_etf_data("data/etf-prices.csv", "AAA")
    normalized, min_val, max_val = normalize(prices)
    
    print(f"Precios originales (Primeros 5):{prices[:5]}")
    print(f"Precios normalizados (Primeros 5):{normalized[:5]}")
    print(f"Min:  {min_val:.2f} | Max: {max_val:.2f}")
    
    print(f"ETF Cargado: {len(prices)} registros")
    print(f"Precio minimo: {prices.min():.2f}")
    print(f"Precio maximo: {prices.max():.2f}")
    print(f"Primeros  5 Precios: {prices[:5]}")
    
    X, y = create_windows(normalized, window_size=60)
    print(f"\nVentanas creadas")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    

 