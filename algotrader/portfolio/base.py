import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class Portfolio:
    """
    Classe représentant un portefeuille d'actifs financiers.
    Gère le capital, les positions et les transactions.
    """

    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialise le portefeuille.

        Args:
            initial_capital: Capital initial du portefeuille
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital  # Capital disponible (non investi)
        self.positions = {}  # Dictionnaire des positions {symbol: quantité}
        self.transactions = []  # Liste des transactions
        self.portfolio_value_history = []  # Historique de la valeur du portefeuille

    def buy(
        self,
        symbol: str,
        quantity: int,
        price: float,
        date: pd.Timestamp,
        commission: float = 0.0,
    ) -> bool:
        """
        Achète une quantité d'un actif.

        Args:
            symbol: Symbole de l'actif
            quantity: Quantité à acheter
            price: Prix d'achat par unité
            date: Date de la transaction
            commission: Commission de la transaction (en pourcentage)

        Returns:
            True si la transaction a réussi, False sinon
        """
        # Calculer le coût total
        commission_amount = price * quantity * commission
        total_cost = price * quantity + commission_amount
        
        # Vérifier si le capital est suffisant
        if total_cost > self.capital:
            return False
        
        # Mettre à jour le capital
        self.capital -= total_cost
        
        # Mettre à jour la position
        if symbol in self.positions:
            self.positions[symbol] += quantity
        else:
            self.positions[symbol] = quantity
        
        # Enregistrer la transaction
        self.transactions.append({
            "date": date,
            "symbol": symbol,
            "action": "buy",
            "quantity": quantity,
            "price": price,
            "commission": commission_amount,
            "total": total_cost,
        })
        
        return True

    def sell(
        self,
        symbol: str,
        quantity: int,
        price: float,
        date: pd.Timestamp,
        commission: float = 0.0,
    ) -> bool:
        """
        Vend une quantité d'un actif.

        Args:
            symbol: Symbole de l'actif
            quantity: Quantité à vendre
            price: Prix de vente par unité
            date: Date de la transaction
            commission: Commission de la transaction (en pourcentage)

        Returns:
            True si la transaction a réussi, False sinon
        """
        # Vérifier si la position existe et est suffisante
        if symbol not in self.positions or self.positions[symbol] < quantity:
            return False
        
        # Calculer le produit de la vente
        commission_amount = price * quantity * commission
        total_proceeds = price * quantity - commission_amount
        
        # Mettre à jour le capital
        self.capital += total_proceeds
        
        # Mettre à jour la position
        self.positions[symbol] -= quantity
        
        # Supprimer la position si elle est nulle
        if self.positions[symbol] == 0:
            del self.positions[symbol]
        
        # Enregistrer la transaction
        self.transactions.append({
            "date": date,
            "symbol": symbol,
            "action": "sell",
            "quantity": quantity,
            "price": price,
            "commission": commission_amount,
            "total": total_proceeds,
        })
        
        return True

    def get_position(self, symbol: str) -> int:
        """
        Récupère la quantité détenue d'un actif.

        Args:
            symbol: Symbole de l'actif

        Returns:
            Quantité détenue (0 si l'actif n'est pas détenu)
        """
        return self.positions.get(symbol, 0)

    def get_positions_value(self, prices: Dict[str, float]) -> float:
        """
        Calcule la valeur totale des positions.

        Args:
            prices: Dictionnaire des prix actuels {symbol: prix}

        Returns:
            Valeur totale des positions
        """
        total_value = 0.0
        
        for symbol, quantity in self.positions.items():
            if symbol in prices:
                total_value += quantity * prices[symbol]
        
        return total_value

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calcule la valeur totale du portefeuille (capital + positions).

        Args:
            prices: Dictionnaire des prix actuels {symbol: prix}

        Returns:
            Valeur totale du portefeuille
        """
        return self.capital + self.get_positions_value(prices)

    def update_portfolio_value(self, date: pd.Timestamp, prices: Dict[str, float]):
        """
        Met à jour l'historique de la valeur du portefeuille.

        Args:
            date: Date de la mise à jour
            prices: Dictionnaire des prix actuels {symbol: prix}
        """
        portfolio_value = self.get_portfolio_value(prices)
        
        self.portfolio_value_history.append({
            "date": date,
            "value": portfolio_value,
        })

    def get_portfolio_value_history(self) -> pd.DataFrame:
        """
        Récupère l'historique de la valeur du portefeuille sous forme de DataFrame.

        Returns:
            DataFrame contenant l'historique de la valeur du portefeuille
        """
        return pd.DataFrame(self.portfolio_value_history).set_index("date")

    def get_transactions(self) -> pd.DataFrame:
        """
        Récupère l'historique des transactions sous forme de DataFrame.

        Returns:
            DataFrame contenant l'historique des transactions
        """
        return pd.DataFrame(self.transactions)

    def reset(self):
        """
        Réinitialise le portefeuille à son état initial.
        """
        self.capital = self.initial_capital
        self.positions = {}
        self.transactions = []
        self.portfolio_value_history = [] 