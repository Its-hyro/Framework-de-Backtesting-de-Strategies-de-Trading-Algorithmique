import pandas as pd
import numpy as np
from typing import Dict, Any
from ..data.indicators import sma, ema
from .base import Strategy


class MovingAverageCrossover(Strategy):
    """
    Stratégie de trading basée sur le croisement de deux moyennes mobiles.
    Génère un signal d'achat lorsque la moyenne mobile rapide croise au-dessus
    de la moyenne mobile lente, et un signal de vente lorsqu'elle croise en dessous.
    """

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 30,
        ma_type: str = "sma",
        column: str = "close",
        name: str = "MovingAverageCrossover",
        position_mode: str = "signal_only"  # Nouveau paramètre
    ):
        """
        Initialise la stratégie de croisement de moyennes mobiles.

        Args:
            short_window: Taille de la fenêtre pour la moyenne mobile rapide
            long_window: Taille de la fenêtre pour la moyenne mobile lente
            ma_type: Type de moyenne mobile ('sma' ou 'ema')
            column: Colonne sur laquelle calculer les moyennes mobiles
            name: Nom de la stratégie
            position_mode: Mode de gestion des positions ('signal_only' ou 'maintain')
                - 'signal_only': Génère des signaux uniquement aux croisements
                - 'maintain': Maintient le signal entre les croisements
        """
        super().__init__(name)
        self.parameters = {
            "short_window": short_window,
            "long_window": long_window,
            "ma_type": ma_type,
            "column": column,
            "position_mode": position_mode,
        }

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare les données en calculant les moyennes mobiles.

        Args:
            data: DataFrame contenant les données financières

        Returns:
            DataFrame avec les moyennes mobiles ajoutées
        """
        prepared_data = data.copy()
        
        # Vérifier que la colonne requise est présente
        column = self.parameters["column"]
        if column not in prepared_data.columns:
            raise ValueError(f"La colonne '{column}' est manquante dans les données")
        
        # Calculer les moyennes mobiles
        ma_type = self.parameters["ma_type"]
        short_window = self.parameters["short_window"]
        long_window = self.parameters["long_window"]
        
        if ma_type == "sma":
            prepared_data["short_ma"] = sma(prepared_data, column, short_window)
            prepared_data["long_ma"] = sma(prepared_data, column, long_window)
        elif ma_type == "ema":
            prepared_data["short_ma"] = ema(prepared_data, column, short_window)
            prepared_data["long_ma"] = ema(prepared_data, column, long_window)
        else:
            raise ValueError(f"Type de moyenne mobile non pris en charge: {ma_type}")
        
        return prepared_data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Génère des signaux de trading basés sur le croisement des moyennes mobiles.

        Args:
            data: DataFrame contenant les données financières

        Returns:
            DataFrame avec une colonne 'signal' ajoutée
        """
        # Préparer les données si elles ne contiennent pas déjà les moyennes mobiles
        if "short_ma" not in data.columns or "long_ma" not in data.columns:
            prepared_data = self.prepare_data(data)
        else:
            prepared_data = data.copy()
        
        # Initialiser la colonne de signal à 0 (pas de signal)
        prepared_data["signal"] = 0
        
        # Calculer les croisements
        # 1 lorsque short_ma croise au-dessus de long_ma
        # -1 lorsque short_ma croise en dessous de long_ma
        prepared_data["signal"] = np.where(
            (prepared_data["short_ma"] > prepared_data["long_ma"]) &
            (prepared_data["short_ma"].shift(1) <= prepared_data["long_ma"].shift(1)),
            1,  # Signal d'achat
            np.where(
                (prepared_data["short_ma"] < prepared_data["long_ma"]) &
                (prepared_data["short_ma"].shift(1) >= prepared_data["long_ma"].shift(1)),
                -1,  # Signal de vente
                0  # Pas de signal
            )
        )
        
        # Si le mode de position est 'maintain', maintenir le signal entre les croisements
        if self.parameters.get("position_mode") == "maintain":
            # Créer une colonne pour la position
            prepared_data["position"] = 0
            
            # Remplir la colonne de position
            for i in range(1, len(prepared_data)):
                if prepared_data["signal"].iloc[i] != 0:
                    # Si un nouveau signal est généré, utiliser ce signal
                    prepared_data.loc[prepared_data.index[i], "position"] = prepared_data["signal"].iloc[i]
                else:
                    # Sinon, maintenir la position précédente
                    prepared_data.loc[prepared_data.index[i], "position"] = prepared_data["position"].iloc[i-1]
            
            # Remplacer la colonne signal par la colonne position
            prepared_data["signal"] = prepared_data["position"].copy()
            
            # Supprimer la colonne position temporaire
            prepared_data = prepared_data.drop("position", axis=1)
        
        return prepared_data


class GoldenCross(MovingAverageCrossover):
    """
    Stratégie de trading basée sur le croisement de la moyenne mobile à 50 jours
    et de la moyenne mobile à 200 jours (Golden Cross).
    """

    def __init__(self, column: str = "close", name: str = "GoldenCross"):
        """
        Initialise la stratégie Golden Cross.

        Args:
            column: Colonne sur laquelle calculer les moyennes mobiles
            name: Nom de la stratégie
        """
        super().__init__(
            short_window=50,
            long_window=200,
            ma_type="sma",
            column=column,
            name=name,
        )


class TripleMovingAverageCrossover(Strategy):
    """
    Stratégie de trading basée sur le croisement de trois moyennes mobiles.
    Génère un signal d'achat lorsque les trois moyennes mobiles sont alignées
    dans l'ordre croissant (court terme > moyen terme > long terme).
    """

    def __init__(
        self,
        short_window: int = 5,
        medium_window: int = 20,
        long_window: int = 50,
        ma_type: str = "sma",
        column: str = "close",
        name: str = "TripleMovingAverageCrossover",
    ):
        """
        Initialise la stratégie de croisement de trois moyennes mobiles.

        Args:
            short_window: Taille de la fenêtre pour la moyenne mobile à court terme
            medium_window: Taille de la fenêtre pour la moyenne mobile à moyen terme
            long_window: Taille de la fenêtre pour la moyenne mobile à long terme
            ma_type: Type de moyenne mobile ('sma' ou 'ema')
            column: Colonne sur laquelle calculer les moyennes mobiles
            name: Nom de la stratégie
        """
        super().__init__(name)
        self.parameters = {
            "short_window": short_window,
            "medium_window": medium_window,
            "long_window": long_window,
            "ma_type": ma_type,
            "column": column,
        }

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare les données en calculant les trois moyennes mobiles.

        Args:
            data: DataFrame contenant les données financières

        Returns:
            DataFrame avec les moyennes mobiles ajoutées
        """
        prepared_data = data.copy()
        
        # Vérifier que la colonne requise est présente
        column = self.parameters["column"]
        if column not in prepared_data.columns:
            raise ValueError(f"La colonne '{column}' est manquante dans les données")
        
        # Calculer les moyennes mobiles
        ma_type = self.parameters["ma_type"]
        short_window = self.parameters["short_window"]
        medium_window = self.parameters["medium_window"]
        long_window = self.parameters["long_window"]
        
        if ma_type == "sma":
            prepared_data["short_ma"] = sma(prepared_data, column, short_window)
            prepared_data["medium_ma"] = sma(prepared_data, column, medium_window)
            prepared_data["long_ma"] = sma(prepared_data, column, long_window)
        elif ma_type == "ema":
            prepared_data["short_ma"] = ema(prepared_data, column, short_window)
            prepared_data["medium_ma"] = ema(prepared_data, column, medium_window)
            prepared_data["long_ma"] = ema(prepared_data, column, long_window)
        else:
            raise ValueError(f"Type de moyenne mobile non pris en charge: {ma_type}")
        
        return prepared_data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Génère des signaux de trading basés sur l'alignement des trois moyennes mobiles.

        Args:
            data: DataFrame contenant les données financières

        Returns:
            DataFrame avec une colonne 'signal' ajoutée
        """
        # Préparer les données si elles ne contiennent pas déjà les moyennes mobiles
        if "short_ma" not in data.columns or "medium_ma" not in data.columns or "long_ma" not in data.columns:
            prepared_data = self.prepare_data(data)
        else:
            prepared_data = data.copy()
        
        # Initialiser la colonne de signal à 0 (pas de signal)
        prepared_data["signal"] = 0
        
        # Générer les signaux
        # Signal d'achat: short_ma > medium_ma > long_ma
        # Signal de vente: short_ma < medium_ma < long_ma
        prepared_data["signal"] = np.where(
            (prepared_data["short_ma"] > prepared_data["medium_ma"]) &
            (prepared_data["medium_ma"] > prepared_data["long_ma"]) &
            (
                (prepared_data["short_ma"].shift(1) <= prepared_data["medium_ma"].shift(1)) |
                (prepared_data["medium_ma"].shift(1) <= prepared_data["long_ma"].shift(1))
            ),
            1,  # Signal d'achat
            np.where(
                (prepared_data["short_ma"] < prepared_data["medium_ma"]) &
                (prepared_data["medium_ma"] < prepared_data["long_ma"]) &
                (
                    (prepared_data["short_ma"].shift(1) >= prepared_data["medium_ma"].shift(1)) |
                    (prepared_data["medium_ma"].shift(1) >= prepared_data["long_ma"].shift(1))
                ),
                -1,  # Signal de vente
                0  # Pas de signal
            )
        )
        
        return prepared_data 