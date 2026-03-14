

import pandas as pd
import numpy as np

def load_etf_data (filepath: str, etf_symbol: str )-> np.ndarray:

    """
    Funcion que carga los valores de un ETF especifico

    Tiene 2 argumentos
    filepath: Ruta al CSV
    etf_symbol: simbolo del ETF (AAA)

    Retorna un array numpy con los precios de cierre
    """
    df = pd.read_csv(filepath)
    etf_data = df[df["fund_symbol"] == etf_symbol]["close"].values

    return etf_data


if __name__ == "__main__":
    prices = load_etf_data("data/etf-prices.csv", "AAA")
    print(f"ETF Cargado: {len(prices)} registros")
    print(f"Precio minimo: {prices.min():.2f}")
    print(f"Precio maximo: {prices.max():.2f}")
    print(f"Primeros  5 Precios: {prices[:5]}")
    