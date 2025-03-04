from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Union, List, Dict, Any


class DataLoader(ABC):
    """
    Classe abstraite de base pour tous les chargeurs de données financières.
    Définit l'interface commune que toutes les implémentations de chargeurs doivent suivre.
    """

    @abstractmethod
    def load(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Charge les données financières pour un symbole donné sur une période spécifiée.

        Args:
            symbol: Le symbole du titre financier (ex: "AAPL" pour Apple)
            start_date: Date de début au format "YYYY-MM-DD"
            end_date: Date de fin au format "YYYY-MM-DD"
            interval: Intervalle de temps entre les données (ex: "1d" pour quotidien, "1h" pour horaire)

        Returns:
            DataFrame pandas contenant les données financières avec au minimum les colonnes:
            - date: Date/heure de la donnée
            - open: Prix d'ouverture
            - high: Prix le plus haut
            - low: Prix le plus bas
            - close: Prix de clôture
            - volume: Volume d'échanges
        """
        pass

    @abstractmethod
    def load_multiple(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Charge les données financières pour plusieurs symboles.

        Args:
            symbols: Liste des symboles à charger
            start_date: Date de début au format "YYYY-MM-DD"
            end_date: Date de fin au format "YYYY-MM-DD"
            interval: Intervalle de temps entre les données

        Returns:
            Dictionnaire avec les symboles comme clés et les DataFrames comme valeurs
        """
        pass

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prétraite les données brutes pour les rendre utilisables par le framework.
        Cette méthode peut être surchargée par les classes dérivées pour des
        prétraitements spécifiques.

        Args:
            data: DataFrame contenant les données brutes

        Returns:
            DataFrame prétraité
        """
        # Vérifier que les colonnes requises sont présentes
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"La colonne '{col}' est manquante dans les données")

        # S'assurer que l'index est bien un DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            if "date" in data.columns:
                # Convertir la colonne date en datetime si nécessaire
                if not pd.api.types.is_datetime64_any_dtype(data["date"]):
                    data["date"] = pd.to_datetime(data["date"])
                data = data.set_index("date")
            else:
                raise ValueError("Aucune colonne 'date' trouvée pour définir l'index")

        # Trier les données par date
        data = data.sort_index()

        # Supprimer les lignes avec des valeurs manquantes
        data = data.dropna()

        return data 