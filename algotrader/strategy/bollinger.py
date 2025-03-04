import pandas as pd
import numpy as np
from typing import Dict, Any
from ..data.indicators import bollinger_bands
from .base import Strategy


class BollingerBandsStrategy(Strategy):
    """
    Stratégie de trading basée sur les bandes de Bollinger.
    Génère un signal d'achat lorsque le prix touche ou dépasse la bande inférieure,
    et un signal de vente lorsque le prix touche ou dépasse la bande supérieure.
    """

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        column: str = "close",
        name: str = "BollingerBandsStrategy",
    ):
        """
        Initialise la stratégie des bandes de Bollinger.

        Args:
            window: Taille de la fenêtre pour la moyenne mobile
            num_std: Nombre d'écarts-types pour les bandes supérieure et inférieure
            column: Colonne sur laquelle calculer les bandes
            name: Nom de la stratégie
        """
        super().__init__(name)
        self.parameters = {
            "window": window,
            "num_std": num_std,
            "column": column,
        }

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare les données en calculant les bandes de Bollinger.

        Args:
            data: DataFrame contenant les données financières

        Returns:
            DataFrame avec les bandes de Bollinger ajoutées
        """
        prepared_data = data.copy()
        
        # Vérifier que la colonne requise est présente
        column = self.parameters["column"]
        if column not in prepared_data.columns:
            raise ValueError(f"La colonne '{column}' est manquante dans les données")
        
        # Calculer les bandes de Bollinger
        window = self.parameters["window"]
        num_std = self.parameters["num_std"]
        bb = bollinger_bands(prepared_data, column, window, num_std)
        
        prepared_data["bb_middle"] = bb["middle_band"]
        prepared_data["bb_upper"] = bb["upper_band"]
        prepared_data["bb_lower"] = bb["lower_band"]
        
        # Calculer la largeur des bandes (utile pour certaines variantes de la stratégie)
        prepared_data["bb_width"] = (prepared_data["bb_upper"] - prepared_data["bb_lower"]) / prepared_data["bb_middle"]
        
        # Calculer le pourcentage B (position relative du prix dans les bandes)
        prepared_data["bb_pct_b"] = (prepared_data[column] - prepared_data["bb_lower"]) / (prepared_data["bb_upper"] - prepared_data["bb_lower"])
        
        return prepared_data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Génère des signaux de trading basés sur les bandes de Bollinger.

        Args:
            data: DataFrame contenant les données financières

        Returns:
            DataFrame avec une colonne 'signal' ajoutée
        """
        # Préparer les données si elles ne contiennent pas déjà les bandes de Bollinger
        if "bb_middle" not in data.columns or "bb_upper" not in data.columns or "bb_lower" not in data.columns:
            prepared_data = self.prepare_data(data)
        else:
            prepared_data = data.copy()
        
        # Initialiser la colonne de signal à 0 (pas de signal)
        prepared_data["signal"] = 0
        
        # Récupérer la colonne de prix
        column = self.parameters["column"]
        
        # Générer les signaux
        # Signal d'achat: le prix touche ou dépasse la bande inférieure et remonte
        # Signal de vente: le prix touche ou dépasse la bande supérieure et redescend
        prepared_data["signal"] = np.where(
            (prepared_data[column] <= prepared_data["bb_lower"]) &
            (prepared_data[column].shift(1) > prepared_data["bb_lower"].shift(1)),
            1,  # Signal d'achat
            np.where(
                (prepared_data[column] >= prepared_data["bb_upper"]) &
                (prepared_data[column].shift(1) < prepared_data["bb_upper"].shift(1)),
                -1,  # Signal de vente
                0  # Pas de signal
            )
        )
        
        return prepared_data


class BollingerBandsReversion(Strategy):
    """
    Stratégie de retour à la moyenne basée sur les bandes de Bollinger.
    Génère un signal d'achat lorsque le prix touche la bande inférieure,
    et un signal de vente lorsque le prix atteint la bande moyenne.
    """

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        column: str = "close",
        name: str = "BollingerBandsReversion",
    ):
        """
        Initialise la stratégie de retour à la moyenne avec les bandes de Bollinger.

        Args:
            window: Taille de la fenêtre pour la moyenne mobile
            num_std: Nombre d'écarts-types pour les bandes supérieure et inférieure
            column: Colonne sur laquelle calculer les bandes
            name: Nom de la stratégie
        """
        super().__init__(name)
        self.parameters = {
            "window": window,
            "num_std": num_std,
            "column": column,
        }
        self.position = 0  # 0: pas de position, 1: position longue, -1: position courte

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare les données en calculant les bandes de Bollinger.

        Args:
            data: DataFrame contenant les données financières

        Returns:
            DataFrame avec les bandes de Bollinger ajoutées
        """
        prepared_data = data.copy()
        
        # Vérifier que la colonne requise est présente
        column = self.parameters["column"]
        if column not in prepared_data.columns:
            raise ValueError(f"La colonne '{column}' est manquante dans les données")
        
        # Calculer les bandes de Bollinger
        window = self.parameters["window"]
        num_std = self.parameters["num_std"]
        bb = bollinger_bands(prepared_data, column, window, num_std)
        
        prepared_data["bb_middle"] = bb["middle_band"]
        prepared_data["bb_upper"] = bb["upper_band"]
        prepared_data["bb_lower"] = bb["lower_band"]
        
        # Calculer le pourcentage B (position relative du prix dans les bandes)
        prepared_data["bb_pct_b"] = (prepared_data[column] - prepared_data["bb_lower"]) / (prepared_data["bb_upper"] - prepared_data["bb_lower"])
        
        return prepared_data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Génère des signaux de trading basés sur la stratégie de retour à la moyenne.

        Args:
            data: DataFrame contenant les données financières

        Returns:
            DataFrame avec une colonne 'signal' ajoutée
        """
        # Préparer les données si elles ne contiennent pas déjà les bandes de Bollinger
        if "bb_middle" not in data.columns or "bb_upper" not in data.columns or "bb_lower" not in data.columns:
            prepared_data = self.prepare_data(data)
        else:
            prepared_data = data.copy()
        
        # Initialiser la colonne de signal à 0 (pas de signal)
        prepared_data["signal"] = 0
        
        # Récupérer la colonne de prix
        column = self.parameters["column"]
        
        # Réinitialiser la position
        self.position = 0
        
        # Générer les signaux en tenant compte de la position actuelle
        for i in range(1, len(prepared_data)):
            # Si pas de position
            if self.position == 0:
                # Signal d'achat: le prix touche ou dépasse la bande inférieure
                if prepared_data[column].iloc[i] <= prepared_data["bb_lower"].iloc[i]:
                    prepared_data["signal"].iloc[i] = 1
                    self.position = 1
                # Signal de vente: le prix touche ou dépasse la bande supérieure
                elif prepared_data[column].iloc[i] >= prepared_data["bb_upper"].iloc[i]:
                    prepared_data["signal"].iloc[i] = -1
                    self.position = -1
            
            # Si position longue
            elif self.position == 1:
                # Signal de vente: le prix atteint la bande moyenne
                if prepared_data[column].iloc[i] >= prepared_data["bb_middle"].iloc[i]:
                    prepared_data["signal"].iloc[i] = -1
                    self.position = 0
            
            # Si position courte
            elif self.position == -1:
                # Signal d'achat: le prix atteint la bande moyenne
                if prepared_data[column].iloc[i] <= prepared_data["bb_middle"].iloc[i]:
                    prepared_data["signal"].iloc[i] = 1
                    self.position = 0
        
        return prepared_data 