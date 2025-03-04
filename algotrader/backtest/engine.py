import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from ..strategy.base import Strategy
from ..portfolio.base import Portfolio


class BacktestResult:
    """
    Classe pour stocker et analyser les résultats d'un backtest.
    """

    def __init__(
        self,
        strategy_name: str,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_capital: float,
        positions: pd.DataFrame,
        portfolio_value: pd.Series,
        trades: pd.DataFrame,
    ):
        """
        Initialise les résultats du backtest.

        Args:
            strategy_name: Nom de la stratégie utilisée
            symbol: Symbole du titre financier
            start_date: Date de début du backtest
            end_date: Date de fin du backtest
            initial_capital: Capital initial
            positions: DataFrame contenant les positions
            portfolio_value: Série contenant la valeur du portefeuille au fil du temps
            trades: DataFrame contenant les transactions
        """
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.positions = positions
        self.portfolio_value = portfolio_value
        self.trades = trades
        
        # Calculer les métriques de performance
        self._calculate_metrics()

    def _calculate_metrics(self):
        """
        Calcule les métriques de performance du backtest.
        """
        # Rendement total
        self.total_return = (self.portfolio_value.iloc[-1] / self.initial_capital) - 1
        
        # Rendement annualisé
        days = (self.portfolio_value.index[-1] - self.portfolio_value.index[0]).days
        self.annual_return = (1 + self.total_return) ** (365 / days) - 1
        
        # Volatilité
        daily_returns = self.portfolio_value.pct_change().dropna()
        self.volatility = daily_returns.std() * np.sqrt(252)  # Annualisée
        
        # Ratio de Sharpe (en supposant un taux sans risque de 0%)
        self.sharpe_ratio = (daily_returns.mean() * 252) / self.volatility if self.volatility != 0 else 0
        
        # Drawdown maximum
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        self.max_drawdown = drawdown.min()
        
        # Nombre de transactions
        self.num_trades = len(self.trades)
        
        # Pourcentage de transactions gagnantes
        if self.num_trades > 0:
            self.win_rate = len(self.trades[self.trades["profit"] > 0]) / self.num_trades
        else:
            self.win_rate = 0
        
        # Profit moyen par transaction
        if self.num_trades > 0:
            self.avg_profit = self.trades["profit"].mean()
        else:
            self.avg_profit = 0

    def summary(self) -> Dict[str, Any]:
        """
        Résume les résultats du backtest.

        Returns:
            Dictionnaire contenant les métriques de performance
        """
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_capital": self.portfolio_value.iloc[-1],
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "avg_profit": self.avg_profit,
        }

    def print_summary(self):
        """
        Affiche un résumé des résultats du backtest.
        """
        summary = self.summary()
        
        print(f"=== Résultats du Backtest: {self.strategy_name} sur {self.symbol} ===")
        print(f"Période: {self.start_date} à {self.end_date}")
        print(f"Capital initial: {summary['initial_capital']:.2f}")
        print(f"Capital final: {summary['final_capital']:.2f}")
        print(f"Rendement total: {summary['total_return']:.2%}")
        print(f"Rendement annualisé: {summary['annual_return']:.2%}")
        print(f"Volatilité annualisée: {summary['volatility']:.2%}")
        print(f"Ratio de Sharpe: {summary['sharpe_ratio']:.2f}")
        print(f"Drawdown maximum: {summary['max_drawdown']:.2%}")
        print(f"Nombre de transactions: {summary['num_trades']}")
        print(f"Taux de réussite: {summary['win_rate']:.2%}")
        print(f"Profit moyen par transaction: {summary['avg_profit']:.2f}")

    def plot_equity_curve(self, benchmark: pd.Series = None, figsize: Tuple[int, int] = (12, 6)):
        """
        Trace la courbe d'équité du portefeuille.

        Args:
            benchmark: Série optionnelle contenant les valeurs d'un benchmark pour comparaison
            figsize: Taille de la figure
        """
        plt.figure(figsize=figsize)
        
        # Tracer la courbe d'équité
        plt.plot(self.portfolio_value.index, self.portfolio_value, label="Portfolio")
        
        # Tracer le benchmark si fourni
        if benchmark is not None:
            # Normaliser le benchmark à la valeur initiale du portefeuille
            benchmark_norm = benchmark / benchmark.iloc[0] * self.initial_capital
            plt.plot(benchmark.index, benchmark_norm, label="Benchmark", alpha=0.7)
        
        plt.title(f"Courbe d'équité: {self.strategy_name} sur {self.symbol}")
        plt.xlabel("Date")
        plt.ylabel("Valeur du portefeuille")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_returns_distribution(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Trace la distribution des rendements quotidiens.

        Args:
            figsize: Taille de la figure
        """
        daily_returns = self.portfolio_value.pct_change().dropna()
        
        plt.figure(figsize=figsize)
        
        # Tracer l'histogramme des rendements
        sns.histplot(daily_returns, kde=True)
        
        plt.title(f"Distribution des rendements quotidiens: {self.strategy_name} sur {self.symbol}")
        plt.xlabel("Rendement quotidien")
        plt.ylabel("Fréquence")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_drawdown(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Trace le drawdown du portefeuille.

        Args:
            figsize: Taille de la figure
        """
        daily_returns = self.portfolio_value.pct_change().dropna()
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        
        plt.figure(figsize=figsize)
        
        # Tracer le drawdown
        plt.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
        plt.plot(drawdown.index, drawdown, color="red", alpha=0.5)
        
        plt.title(f"Drawdown: {self.strategy_name} sur {self.symbol}")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_trades(self, price_data: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)):
        """
        Trace les transactions sur le graphique des prix.

        Args:
            price_data: DataFrame contenant les données de prix
            figsize: Taille de la figure
        """
        plt.figure(figsize=figsize)
        
        # Tracer le prix
        plt.plot(price_data.index, price_data["close"], label="Prix", alpha=0.7)
        
        # Tracer les achats
        buy_trades = self.trades[self.trades["action"] == "buy"]
        plt.scatter(buy_trades["date"], buy_trades["price"], marker="^", color="green", s=100, label="Achat")
        
        # Tracer les ventes
        sell_trades = self.trades[self.trades["action"] == "sell"]
        plt.scatter(sell_trades["date"], sell_trades["price"], marker="v", color="red", s=100, label="Vente")
        
        plt.title(f"Transactions: {self.strategy_name} sur {self.symbol}")
        plt.xlabel("Date")
        plt.ylabel("Prix")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()


class Backtester:
    """
    Classe pour exécuter des backtests de stratégies de trading.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        portfolio: Portfolio,
        commission: float = 0.001,
        slippage: float = 0.0,
    ):
        """
        Initialise le backtester.

        Args:
            data: DataFrame contenant les données financières
            strategy: Stratégie de trading à tester
            portfolio: Portefeuille à utiliser pour le backtest
            commission: Taux de commission par transaction (en pourcentage)
            slippage: Glissement de prix par transaction (en pourcentage)
        """
        self.data = data.copy()
        self.strategy = strategy
        self.portfolio = portfolio
        self.commission = commission
        self.slippage = slippage
        
        # Vérifier que les données ont un index de type DatetimeIndex
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Les données doivent avoir un index de type DatetimeIndex")
        
        # Vérifier que les colonnes requises sont présentes
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"La colonne '{col}' est manquante dans les données")
        
        # Stocker les dates de début et de fin
        self.start_date = self.data.index[0].strftime("%Y-%m-%d")
        self.end_date = self.data.index[-1].strftime("%Y-%m-%d")
        
        # Stocker le symbole
        self.symbol = self.data["symbol"].iloc[0] if "symbol" in self.data.columns else "UNKNOWN"

    def run(self) -> BacktestResult:
        """
        Exécute le backtest.

        Returns:
            Résultats du backtest
        """
        # Générer les signaux de trading
        signals = self.strategy.generate_signals(self.data)
        
        # Vérifier si des signaux ont été générés
        signal_count = (signals["signal"] != 0).sum()
        print(f"Nombre de signaux générés: {signal_count}")
        
        # Initialiser les structures de données pour suivre les positions et les transactions
        positions = pd.DataFrame(index=signals.index)
        positions["position"] = 0  # Nombre d'unités détenues
        positions["price"] = signals["close"]  # Prix de clôture
        
        # Initialiser le portefeuille
        portfolio_value = pd.Series(index=signals.index)
        portfolio_value.iloc[0] = self.portfolio.capital
        
        # Initialiser la liste des transactions
        trades = []
        
        # Simuler les transactions
        for i in range(1, len(signals)):
            # Récupérer le signal actuel
            current_signal = signals["signal"].iloc[i]
            
            # Récupérer la position actuelle
            current_position = positions["position"].iloc[i-1]
            
            # Calculer la nouvelle position
            new_position = current_position
            
            # Traiter le signal
            if current_signal == 1 and current_position <= 0:  # Signal d'achat
                # Calculer le prix d'achat avec slippage
                buy_price = signals["close"].iloc[i] * (1 + self.slippage)
                
                # Calculer le nombre d'unités à acheter
                available_capital = self.portfolio.capital
                units_to_buy = int(available_capital / buy_price)
                
                if units_to_buy > 0:
                    # Mettre à jour la position
                    new_position = units_to_buy
                    
                    # Calculer le coût total avec commission
                    cost = units_to_buy * buy_price * (1 + self.commission)
                    
                    # Mettre à jour le capital
                    self.portfolio.capital -= cost
                    
                    # Enregistrer la transaction
                    trades.append({
                        "date": signals.index[i],
                        "action": "buy",
                        "price": buy_price,
                        "units": units_to_buy,
                        "cost": cost,
                        "commission": units_to_buy * buy_price * self.commission,
                        "profit": 0,
                    })
                    print(f"Achat à {signals.index[i]}: {units_to_buy} unités à {buy_price}")
            
            elif current_signal == -1 and current_position >= 0:  # Signal de vente
                if current_position > 0:
                    # Calculer le prix de vente avec slippage
                    sell_price = signals["close"].iloc[i] * (1 - self.slippage)
                    
                    # Calculer le produit de la vente
                    proceeds = current_position * sell_price * (1 - self.commission)
                    
                    # Calculer le profit
                    cost_basis = 0
                    for trade in reversed(trades):
                        if trade["action"] == "buy":
                            cost_basis = trade["price"]
                            break
                    
                    profit = (sell_price - cost_basis) * current_position - \
                             (current_position * sell_price * self.commission)
                    
                    # Mettre à jour le capital
                    self.portfolio.capital += proceeds
                    
                    # Mettre à jour la position
                    new_position = 0
                    
                    # Enregistrer la transaction
                    trades.append({
                        "date": signals.index[i],
                        "action": "sell",
                        "price": sell_price,
                        "units": current_position,
                        "cost": 0,
                        "commission": current_position * sell_price * self.commission,
                        "profit": profit,
                    })
                    print(f"Vente à {signals.index[i]}: {current_position} unités à {sell_price}")
            
            # Mettre à jour la position (en utilisant loc au lieu de iloc pour éviter le warning)
            positions.loc[signals.index[i], "position"] = new_position
            
            # Calculer la valeur du portefeuille
            portfolio_value.iloc[i] = self.portfolio.capital + (new_position * signals["close"].iloc[i])
        
        # Convertir la liste des transactions en DataFrame
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["date", "action", "price", "units", "cost", "commission", "profit"])
        
        # Créer et retourner les résultats du backtest
        return BacktestResult(
            strategy_name=self.strategy.name,
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.portfolio.initial_capital,
            positions=positions,
            portfolio_value=portfolio_value,
            trades=trades_df,
        ) 