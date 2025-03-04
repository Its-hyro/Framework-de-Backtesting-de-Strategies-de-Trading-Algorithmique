from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple


class Strategy(ABC):
    """
    Classe abstraite de base pour toutes les stratégies de trading.
    Définit l'interface commune que toutes les implémentations de stratégies doivent suivre.
    """

    def __init__(self, name: str = None):
        """
        Initialise la stratégie.

        Args:
            name: Nom optionnel de la stratégie
        """
        self.name = name or self.__class__.__name__
        self.parameters = {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Génère des signaux de trading basés sur les données fournies.

        Args:
            data: DataFrame contenant les données financières avec au minimum les colonnes OHLCV

        Returns:
            DataFrame avec une colonne 'signal' ajoutée:
            - 1 pour un signal d'achat
            - -1 pour un signal de vente
            - 0 pour aucun signal
        """
        pass

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Définit les paramètres de la stratégie.

        Args:
            parameters: Dictionnaire des paramètres à définir
        """
        self.parameters.update(parameters)

    def get_parameters(self) -> Dict[str, Any]:
        """
        Récupère les paramètres actuels de la stratégie.

        Returns:
            Dictionnaire des paramètres
        """
        return self.parameters.copy()

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare les données pour la stratégie en ajoutant des indicateurs ou en effectuant
        d'autres transformations nécessaires.

        Cette méthode peut être surchargée par les classes dérivées pour des
        prétraitements spécifiques.

        Args:
            data: DataFrame contenant les données brutes

        Returns:
            DataFrame préparé pour la stratégie
        """
        return data

    def __str__(self) -> str:
        """
        Représentation sous forme de chaîne de caractères de la stratégie.

        Returns:
            Chaîne de caractères décrivant la stratégie et ses paramètres
        """
        params_str = ", ".join([f"{k}={v}" for k, v in self.parameters.items()])
        return f"{self.name}({params_str})"


class CombinedStrategy(Strategy):
    """
    Stratégie qui combine plusieurs stratégies individuelles.
    """

    def __init__(self, strategies: List[Strategy], name: str = "CombinedStrategy"):
        """
        Initialise la stratégie combinée.

        Args:
            strategies: Liste des stratégies à combiner
            name: Nom de la stratégie combinée
        """
        super().__init__(name)
        self.strategies = strategies

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Génère des signaux de trading en combinant les signaux de toutes les stratégies.

        Args:
            data: DataFrame contenant les données financières

        Returns:
            DataFrame avec une colonne 'signal' ajoutée
        """
        # Préparer les données
        prepared_data = self.prepare_data(data)
        
        # Générer les signaux pour chaque stratégie
        signals = pd.DataFrame(index=prepared_data.index)
        
        for i, strategy in enumerate(self.strategies):
            strategy_signals = strategy.generate_signals(prepared_data)
            signals[f"signal_{i}"] = strategy_signals["signal"]
        
        # Combiner les signaux (par défaut, prendre la moyenne)
        signals["signal"] = signals.mean(axis=1).round()
        
        # Ajouter les signaux au DataFrame original
        result = data.copy()
        result["signal"] = signals["signal"]
        
        return result

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare les données pour toutes les stratégies.

        Args:
            data: DataFrame contenant les données brutes

        Returns:
            DataFrame préparé pour les stratégies
        """
        prepared_data = data.copy()
        
        for strategy in self.strategies:
            prepared_data = strategy.prepare_data(prepared_data)
        
        return prepared_data

    def __str__(self) -> str:
        """
        Représentation sous forme de chaîne de caractères de la stratégie combinée.

        Returns:
            Chaîne de caractères décrivant la stratégie combinée
        """
        strategies_str = ", ".join([str(s) for s in self.strategies])
        return f"{self.name}([{strategies_str}])" 