import pandas as pd
import os
import time
from typing import List, Dict
from alpha_vantage.timeseries import TimeSeries
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base import DataLoader


class AlphaVantageDataLoader(DataLoader):
    """
    Implémentation du chargeur de données utilisant l'API Alpha Vantage.
    """

    def __init__(self, api_key: str = None, cache_dir: str = None):
        """
        Initialise le chargeur de données Alpha Vantage.

        Args:
            api_key: Clé API Alpha Vantage. Si None, la clé sera recherchée dans
                    la variable d'environnement ALPHA_VANTAGE_API_KEY.
            cache_dir: Répertoire optionnel pour mettre en cache les données téléchargées
        """
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Aucune clé API Alpha Vantage fournie. Veuillez fournir une clé API "
                "ou définir la variable d'environnement ALPHA_VANTAGE_API_KEY."
            )
        self.cache_dir = cache_dir
        self.ts = TimeSeries(key=self.api_key, output_format="pandas")
        self.request_limit = 5  # Limite de requêtes par minute pour le plan gratuit
        self.request_count = 0
        self.last_request_time = 0

    def _manage_rate_limit(self):
        """
        Gère la limite de taux de l'API Alpha Vantage en attendant si nécessaire.
        """
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # Si nous avons atteint la limite de requêtes et que moins d'une minute s'est écoulée
        if self.request_count >= self.request_limit and elapsed < 60:
            # Attendre le temps nécessaire pour compléter une minute
            sleep_time = 60 - elapsed
            print(f"Limite de taux atteinte, attente de {sleep_time:.2f} secondes...")
            time.sleep(sleep_time)
            self.request_count = 0
            self.last_request_time = time.time()
        # Si plus d'une minute s'est écoulée depuis la dernière requête, réinitialiser le compteur
        elif elapsed >= 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        # Incrémenter le compteur de requêtes
        self.request_count += 1
        if self.request_count == 1:
            self.last_request_time = current_time

    def load(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Charge les données financières pour un symbole donné depuis Alpha Vantage.

        Args:
            symbol: Le symbole du titre financier (ex: "AAPL" pour Apple)
            start_date: Date de début au format "YYYY-MM-DD"
            end_date: Date de fin au format "YYYY-MM-DD"
            interval: Intervalle de temps entre les données (ex: "1d" pour quotidien, "1h" pour horaire)

        Returns:
            DataFrame pandas contenant les données financières
        """
        try:
            # Gérer la limite de taux
            self._manage_rate_limit()
            
            # Convertir l'intervalle au format Alpha Vantage
            av_interval = self._convert_interval(interval)
            
            # Télécharger les données depuis Alpha Vantage
            if av_interval == "daily":
                data, meta_data = self.ts.get_daily(symbol=symbol, outputsize="full")
            elif av_interval == "weekly":
                data, meta_data = self.ts.get_weekly(symbol=symbol)
            elif av_interval == "monthly":
                data, meta_data = self.ts.get_monthly(symbol=symbol)
            elif av_interval.startswith("intraday"):
                # Extraire l'intervalle intraday (ex: "1min", "5min", "15min", "30min", "60min")
                intraday_interval = av_interval.split("_")[1]
                data, meta_data = self.ts.get_intraday(
                    symbol=symbol, interval=intraday_interval, outputsize="full"
                )
            else:
                raise ValueError(f"Intervalle non pris en charge: {interval}")
            
            # Renommer les colonnes pour correspondre à notre format standard
            data.columns = [col.split(". ")[1].lower() for col in data.columns]
            
            # Filtrer les données selon les dates de début et de fin
            data = data.loc[start_date:end_date]
            
            # Prétraiter les données
            data = self.preprocess(data)
            
            # Ajouter une colonne pour le symbole
            data["symbol"] = symbol
            
            return data
            
        except Exception as e:
            raise Exception(f"Erreur lors du chargement des données pour {symbol}: {str(e)}")

    def load_multiple(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Charge les données financières pour plusieurs symboles en séquentiel
        pour respecter les limites de taux d'Alpha Vantage.

        Args:
            symbols: Liste des symboles à charger
            start_date: Date de début au format "YYYY-MM-DD"
            end_date: Date de fin au format "YYYY-MM-DD"
            interval: Intervalle de temps entre les données

        Returns:
            Dictionnaire avec les symboles comme clés et les DataFrames comme valeurs
        """
        result = {}
        
        for symbol in symbols:
            try:
                data = self.load(symbol, start_date, end_date, interval)
                result[symbol] = data
            except Exception as e:
                print(f"Erreur lors du chargement des données pour {symbol}: {str(e)}")
        
        return result

    def _convert_interval(self, interval: str) -> str:
        """
        Convertit l'intervalle au format utilisé par Alpha Vantage.

        Args:
            interval: Intervalle au format du framework (ex: "1d", "1h", "15m")

        Returns:
            Intervalle au format Alpha Vantage
        """
        # Mapping des intervalles
        interval_map = {
            "1d": "daily",
            "1w": "weekly",
            "1mo": "monthly",
            "1m": "intraday_1min",
            "5m": "intraday_5min",
            "15m": "intraday_15min",
            "30m": "intraday_30min",
            "1h": "intraday_60min",
        }
        
        if interval in interval_map:
            return interval_map[interval]
        else:
            raise ValueError(
                f"Intervalle non pris en charge: {interval}. "
                f"Les intervalles pris en charge sont: {list(interval_map.keys())}"
            ) 