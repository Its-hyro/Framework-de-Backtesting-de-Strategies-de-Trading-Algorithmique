import pandas as pd
import numpy as np
from typing import Union, Optional


def sma(data: pd.DataFrame, column: str = "close", window: int = 20) -> pd.Series:
    """
    Calcule la moyenne mobile simple (Simple Moving Average).

    Args:
        data: DataFrame contenant les données
        column: Nom de la colonne sur laquelle calculer la SMA
        window: Taille de la fenêtre pour la moyenne mobile

    Returns:
        Série pandas contenant la SMA
    """
    return data[column].rolling(window=window).mean()


def ema(data: pd.DataFrame, column: str = "close", window: int = 20) -> pd.Series:
    """
    Calcule la moyenne mobile exponentielle (Exponential Moving Average).

    Args:
        data: DataFrame contenant les données
        column: Nom de la colonne sur laquelle calculer l'EMA
        window: Taille de la fenêtre pour la moyenne mobile

    Returns:
        Série pandas contenant l'EMA
    """
    return data[column].ewm(span=window, adjust=False).mean()


def bollinger_bands(
    data: pd.DataFrame, column: str = "close", window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """
    Calcule les bandes de Bollinger.

    Args:
        data: DataFrame contenant les données
        column: Nom de la colonne sur laquelle calculer les bandes
        window: Taille de la fenêtre pour la moyenne mobile
        num_std: Nombre d'écarts-types pour les bandes supérieure et inférieure

    Returns:
        DataFrame contenant les colonnes 'middle_band', 'upper_band', 'lower_band'
    """
    middle_band = sma(data, column, window)
    std = data[column].rolling(window=window).std()
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return pd.DataFrame({
        "middle_band": middle_band,
        "upper_band": upper_band,
        "lower_band": lower_band
    })


def macd(
    data: pd.DataFrame,
    column: str = "close",
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Calcule le MACD (Moving Average Convergence Divergence).

    Args:
        data: DataFrame contenant les données
        column: Nom de la colonne sur laquelle calculer le MACD
        fast_period: Période pour la moyenne mobile rapide
        slow_period: Période pour la moyenne mobile lente
        signal_period: Période pour la ligne de signal

    Returns:
        DataFrame contenant les colonnes 'macd', 'signal', 'histogram'
    """
    fast_ema = ema(data, column, fast_period)
    slow_ema = ema(data, column, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    })


def rsi(data: pd.DataFrame, column: str = "close", window: int = 14) -> pd.Series:
    """
    Calcule l'indice de force relative (Relative Strength Index).

    Args:
        data: DataFrame contenant les données
        column: Nom de la colonne sur laquelle calculer le RSI
        window: Taille de la fenêtre pour le RSI

    Returns:
        Série pandas contenant le RSI
    """
    delta = data[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Pour éviter la division par zéro
    avg_loss = avg_loss.replace(0, np.finfo(float).eps)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calcule l'Average True Range (ATR).

    Args:
        data: DataFrame contenant les données (doit avoir les colonnes 'high', 'low', 'close')
        window: Taille de la fenêtre pour l'ATR

    Returns:
        Série pandas contenant l'ATR
    """
    high = data["high"]
    low = data["low"]
    close = data["close"]
    
    # Calculer le True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.DataFrame({
        "tr1": tr1,
        "tr2": tr2,
        "tr3": tr3
    }).max(axis=1)
    
    # Calculer l'ATR
    atr = tr.rolling(window=window).mean()
    
    return atr


def stochastic_oscillator(
    data: pd.DataFrame, k_window: int = 14, d_window: int = 3
) -> pd.DataFrame:
    """
    Calcule l'oscillateur stochastique.

    Args:
        data: DataFrame contenant les données (doit avoir les colonnes 'high', 'low', 'close')
        k_window: Taille de la fenêtre pour %K
        d_window: Taille de la fenêtre pour %D

    Returns:
        DataFrame contenant les colonnes '%K' et '%D'
    """
    low_min = data["low"].rolling(window=k_window).min()
    high_max = data["high"].rolling(window=k_window).max()
    
    # Calculer %K
    k = 100 * ((data["close"] - low_min) / (high_max - low_min))
    
    # Calculer %D (moyenne mobile de %K)
    d = k.rolling(window=d_window).mean()
    
    return pd.DataFrame({
        "%K": k,
        "%D": d
    })


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute plusieurs indicateurs techniques au DataFrame.

    Args:
        data: DataFrame contenant les données OHLCV

    Returns:
        DataFrame avec les indicateurs ajoutés
    """
    # Créer une copie pour éviter de modifier l'original
    result = data.copy()
    
    # Ajouter les moyennes mobiles
    result["sma_20"] = sma(data, window=20)
    result["sma_50"] = sma(data, window=50)
    result["sma_200"] = sma(data, window=200)
    result["ema_20"] = ema(data, window=20)
    
    # Ajouter les bandes de Bollinger
    bb = bollinger_bands(data, window=20)
    result["bb_middle"] = bb["middle_band"]
    result["bb_upper"] = bb["upper_band"]
    result["bb_lower"] = bb["lower_band"]
    
    # Ajouter le MACD
    macd_data = macd(data)
    result["macd"] = macd_data["macd"]
    result["macd_signal"] = macd_data["signal"]
    result["macd_hist"] = macd_data["histogram"]
    
    # Ajouter le RSI
    result["rsi_14"] = rsi(data, window=14)
    
    # Ajouter l'ATR
    result["atr_14"] = atr(data, window=14)
    
    # Ajouter l'oscillateur stochastique
    stoch = stochastic_oscillator(data)
    result["%K"] = stoch["%K"]
    result["%D"] = stoch["%D"]
    
    return result 