import pandas as pd
import numpy as np
from typing import Dict, Any
from ..data.indicators import rsi
from .base import Strategy


class RSIStrategy(Strategy):
    """
    Stratégie de trading basée sur l'indice de force relative (RSI).
    Génère un signal d'achat lorsque le RSI passe en dessous d'un seuil de survente
    et remonte, et un signal de vente lorsque le RSI passe au-dessus d'un seuil de
    surachat et redescend.
    """

    def __init__(
        self,
        window: int = 14,
        oversold_threshold: int = 30,
        overbought_threshold: int = 70,
        column: str = "close",
        name: str = "RSIStrategy",
    ):
        """
        Initialise la stratégie RSI.

        Args:
            window: Taille de la fenêtre pour le calcul du RSI
            oversold_threshold: Seuil de survente (typiquement 30)
            overbought_threshold: Seuil de surachat (typiquement 70)
            column: Colonne sur laquelle calculer le RSI
            name: Nom de la stratégie
        """
        super().__init__(name)
        self.parameters = {
            "window": window,
            "oversold_threshold": oversold_threshold,
            "overbought_threshold": overbought_threshold,
            "column": column,
        }

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare les données en calculant le RSI.

        Args:
            data: DataFrame contenant les données financières

        Returns:
            DataFrame avec le RSI ajouté
        """
        prepared_data = data.copy()
        
        # Vérifier que la colonne requise est présente
        column = self.parameters["column"]
        if column not in prepared_data.columns:
            raise ValueError(f"La colonne '{column}' est manquante dans les données")
        
        # Calculer le RSI
        window = self.parameters["window"]
        prepared_data["rsi"] = rsi(prepared_data, column, window)
        
        return prepared_data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Génère des signaux de trading basés sur le RSI.

        Args:
            data: DataFrame contenant les données financières

        Returns:
            DataFrame avec une colonne 'signal' ajoutée
        """
        # Préparer les données si elles ne contiennent pas déjà le RSI
        if "rsi" not in data.columns:
            prepared_data = self.prepare_data(data)
        else:
            prepared_data = data.copy()
        
        # Initialiser la colonne de signal à 0 (pas de signal)
        prepared_data["signal"] = 0
        
        # Récupérer les paramètres
        oversold = self.parameters["oversold_threshold"]
        overbought = self.parameters["overbought_threshold"]
        
        # Générer les signaux
        # Signal d'achat: RSI remonte après être passé sous le seuil de survente
        # Signal de vente: RSI redescend après être passé au-dessus du seuil de surachat
        prepared_data["signal"] = np.where(
            (prepared_data["rsi"] > oversold) &
            (prepared_data["rsi"].shift(1) <= oversold),
            1,  # Signal d'achat
            np.where(
                (prepared_data["rsi"] < overbought) &
                (prepared_data["rsi"].shift(1) >= overbought),
                -1,  # Signal de vente
                0  # Pas de signal
            )
        )
        
        return prepared_data


class RSIDivergenceStrategy(Strategy):
    """
    Stratégie de trading basée sur la divergence du RSI avec le prix.
    Détecte les divergences entre le RSI et le prix pour générer des signaux.
    """

    def __init__(
        self,
        window: int = 14,
        divergence_window: int = 20,
        column: str = "close",
        name: str = "RSIDivergenceStrategy",
    ):
        """
        Initialise la stratégie de divergence RSI.

        Args:
            window: Taille de la fenêtre pour le calcul du RSI
            divergence_window: Taille de la fenêtre pour détecter les divergences
            column: Colonne sur laquelle calculer le RSI
            name: Nom de la stratégie
        """
        super().__init__(name)
        self.parameters = {
            "window": window,
            "divergence_window": divergence_window,
            "column": column,
        }

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare les données en calculant le RSI.

        Args:
            data: DataFrame contenant les données financières

        Returns:
            DataFrame avec le RSI ajouté
        """
        prepared_data = data.copy()
        
        # Vérifier que la colonne requise est présente
        column = self.parameters["column"]
        if column not in prepared_data.columns:
            raise ValueError(f"La colonne '{column}' est manquante dans les données")
        
        # Calculer le RSI
        window = self.parameters["window"]
        prepared_data["rsi"] = rsi(prepared_data, column, window)
        
        return prepared_data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Génère des signaux de trading basés sur la divergence du RSI.

        Args:
            data: DataFrame contenant les données financières

        Returns:
            DataFrame avec une colonne 'signal' ajoutée
        """
        # Préparer les données si elles ne contiennent pas déjà le RSI
        if "rsi" not in data.columns:
            prepared_data = self.prepare_data(data)
        else:
            prepared_data = data.copy()
        
        # Initialiser la colonne de signal à 0 (pas de signal)
        prepared_data["signal"] = 0
        
        # Récupérer les paramètres
        column = self.parameters["column"]
        divergence_window = self.parameters["divergence_window"]
        
        # Calculer les extremums locaux du prix et du RSI
        prepared_data["price_high"] = prepared_data[column].rolling(window=divergence_window, center=True).apply(
            lambda x: 1 if x.iloc[len(x)//2] == max(x) else 0
        )
        prepared_data["price_low"] = prepared_data[column].rolling(window=divergence_window, center=True).apply(
            lambda x: 1 if x.iloc[len(x)//2] == min(x) else 0
        )
        prepared_data["rsi_high"] = prepared_data["rsi"].rolling(window=divergence_window, center=True).apply(
            lambda x: 1 if x.iloc[len(x)//2] == max(x) else 0
        )
        prepared_data["rsi_low"] = prepared_data["rsi"].rolling(window=divergence_window, center=True).apply(
            lambda x: 1 if x.iloc[len(x)//2] == min(x) else 0
        )
        
        # Détecter les divergences
        # Divergence baissière: nouveau sommet du prix mais pas du RSI
        # Divergence haussière: nouveau creux du prix mais pas du RSI
        prepared_data["signal"] = np.where(
            (prepared_data["price_high"] == 1) & (prepared_data["rsi_high"] == 0) &
            (prepared_data[column] > prepared_data[column].shift(divergence_window)) &
            (prepared_data["rsi"] < prepared_data["rsi"].shift(divergence_window)),
            -1,  # Signal de vente (divergence baissière)
            np.where(
                (prepared_data["price_low"] == 1) & (prepared_data["rsi_low"] == 0) &
                (prepared_data[column] < prepared_data[column].shift(divergence_window)) &
                (prepared_data["rsi"] > prepared_data["rsi"].shift(divergence_window)),
                1,  # Signal d'achat (divergence haussière)
                0  # Pas de signal
            )
        )
        
        # Nettoyer les colonnes temporaires
        prepared_data = prepared_data.drop(["price_high", "price_low", "rsi_high", "rsi_low"], axis=1)
        
        return prepared_data 