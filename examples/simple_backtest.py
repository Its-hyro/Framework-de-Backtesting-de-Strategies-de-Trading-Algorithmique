import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algotrader.data import YahooFinanceDataLoader, add_indicators
from algotrader.strategy import MovingAverageCrossover, RSIStrategy, BollingerBandsStrategy, CombinedStrategy
from algotrader.portfolio import Portfolio
from algotrader.backtest import Backtester
from algotrader.visualization import plot_price_with_signals, create_performance_tearsheet


def run_simple_backtest():
    """
    Exécute un backtest simple avec une stratégie de croisement de moyennes mobiles.
    """
    # Charger les données
    print("Chargement des données...")
    data_loader = YahooFinanceDataLoader()
    
    # Liste des symboles à essayer
    symbols_to_try = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    
    # Essayer de charger les données pour chaque symbole jusqu'à ce qu'un fonctionne
    data = None
    for symbol in symbols_to_try:
        try:
            print(f"Tentative de chargement des données pour {symbol}...")
            data = data_loader.load(
                symbol=symbol,
                start_date="2020-01-01",
                end_date="2022-01-01",
                interval="1d",
            )
            print(f"Données chargées avec succès pour {symbol}")
            break
        except Exception as e:
            print(f"Échec du chargement des données pour {symbol}: {str(e)}")
    
    # Si aucun symbole n'a fonctionné, utiliser des données de démonstration
    if data is None:
        print("Impossible de charger des données réelles. Utilisation de données de démonstration...")
        # Créer des dates de démonstration
        dates = pd.date_range(start="2020-01-01", end="2022-01-01", freq='D')
        
        # Créer un prix de base avec une tendance haussière et de la volatilité
        np.random.seed(42)  # Pour la reproductibilité
        n = len(dates)
        
        # Tendance de base (croissance linéaire)
        base_trend = np.linspace(100, 200, n)
        
        # Ajouter de la volatilité (mouvement aléatoire)
        volatility = np.random.normal(0, 3, n)
        cumulative_volatility = np.cumsum(volatility)
        
        # Combiner tendance et volatilité
        close_prices = base_trend + cumulative_volatility
        
        # Créer les autres colonnes avec une volatilité réaliste
        data = pd.DataFrame({
            'open': close_prices * (1 + np.random.normal(0, 0.005, n)),
            'high': close_prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
            'low': close_prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
            'close': close_prices,
            'volume': np.random.randint(500000, 1500000, n),
            'symbol': ["DEMO" for _ in range(n)]
        })
        
        # Définir la date comme index
        data.index = dates
    
    # Ajouter les indicateurs techniques
    print("Ajout des indicateurs techniques...")
    data_with_indicators = add_indicators(data)
    
    # Créer une stratégie
    print("Création de la stratégie...")
    strategy = MovingAverageCrossover(short_window=10, long_window=30, position_mode="maintain")
    
    # Initialiser le portefeuille
    print("Initialisation du portefeuille...")
    portfolio = Portfolio(initial_capital=10000)
    
    # Exécuter le backtest
    print("Exécution du backtest...")
    backtester = Backtester(data_with_indicators, strategy, portfolio)
    results = backtester.run()
    
    # Afficher les résultats
    print("\nRésultats du backtest:")
    results.print_summary()
    
    # Tracer les graphiques
    print("\nAffichage des graphiques...")
    
    # Tracer le prix avec les signaux
    signals = strategy.generate_signals(data_with_indicators)
    plot_price_with_signals(data_with_indicators, signals)
    
    # Créer une feuille de performance
    create_performance_tearsheet(results)
    
    return results


def compare_strategies():
    """
    Compare plusieurs stratégies de trading.
    """
    # Charger les données
    print("Chargement des données...")
    data_loader = YahooFinanceDataLoader()
    
    # Liste des symboles à essayer
    symbols_to_try = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    
    # Essayer de charger les données pour chaque symbole jusqu'à ce qu'un fonctionne
    data = None
    for symbol in symbols_to_try:
        try:
            print(f"Tentative de chargement des données pour {symbol}...")
            data = data_loader.load(
                symbol=symbol,
                start_date="2020-01-01",
                end_date="2022-01-01",
                interval="1d",
            )
            print(f"Données chargées avec succès pour {symbol}")
            break
        except Exception as e:
            print(f"Échec du chargement des données pour {symbol}: {str(e)}")
    
    # Si aucun symbole n'a fonctionné, utiliser des données de démonstration
    if data is None:
        print("Impossible de charger des données réelles. Utilisation de données de démonstration...")
        # Créer des dates de démonstration
        dates = pd.date_range(start="2020-01-01", end="2022-01-01", freq='D')
        
        # Créer un prix de base avec une tendance haussière et de la volatilité
        np.random.seed(42)  # Pour la reproductibilité
        n = len(dates)
        
        # Tendance de base (croissance linéaire)
        base_trend = np.linspace(100, 200, n)
        
        # Ajouter de la volatilité (mouvement aléatoire)
        volatility = np.random.normal(0, 3, n)
        cumulative_volatility = np.cumsum(volatility)
        
        # Combiner tendance et volatilité
        close_prices = base_trend + cumulative_volatility
        
        # Créer les autres colonnes avec une volatilité réaliste
        data = pd.DataFrame({
            'open': close_prices * (1 + np.random.normal(0, 0.005, n)),
            'high': close_prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
            'low': close_prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
            'close': close_prices,
            'volume': np.random.randint(500000, 1500000, n),
            'symbol': ["DEMO" for _ in range(n)]
        })
        
        # Définir la date comme index
        data.index = dates
    
    # Ajouter les indicateurs techniques
    print("Ajout des indicateurs techniques...")
    data_with_indicators = add_indicators(data)
    
    # Créer les stratégies
    print("Création des stratégies...")
    ma_strategy = MovingAverageCrossover(short_window=10, long_window=30, position_mode="maintain")
    rsi_strategy = RSIStrategy(window=10, oversold_threshold=30, overbought_threshold=70)
    bb_strategy = BollingerBandsStrategy(window=15, num_std=2.0)
    
    # Créer une stratégie combinée
    combined_strategy = CombinedStrategy(
        strategies=[ma_strategy, rsi_strategy, bb_strategy],
        name="StratégieCombinée",
    )
    
    # Initialiser le portefeuille
    print("Initialisation du portefeuille...")
    portfolio = Portfolio(initial_capital=10000)
    
    # Liste pour stocker les résultats
    results_list = []
    
    # Exécuter les backtests pour chaque stratégie
    for strategy in [ma_strategy, rsi_strategy, bb_strategy, combined_strategy]:
        print(f"\nExécution du backtest pour {strategy.name}...")
        
        # Réinitialiser le portefeuille
        portfolio.reset()
        
        # Exécuter le backtest
        backtester = Backtester(data_with_indicators, strategy, portfolio)
        results = backtester.run()
        
        # Afficher les résultats
        print(f"\nRésultats du backtest pour {strategy.name}:")
        results.print_summary()
        
        # Stocker les résultats
        results_list.append(results)
    
    # Comparer les stratégies
    print("\nComparaison des stratégies...")
    from algotrader.analysis import compare_strategies, plot_equity_curves
    
    # Créer un benchmark (buy and hold)
    benchmark = data["close"]
    
    # Comparer les métriques de performance
    metrics_df = compare_strategies(results_list, benchmark)
    print("\nMétriques de performance:")
    print(metrics_df)
    
    # Tracer les courbes d'équité
    plot_equity_curves(results_list, benchmark)
    
    return results_list, metrics_df


if __name__ == "__main__":
    # Exécuter un backtest simple
    print("=== Exécution d'un backtest simple ===\n")
    results = run_simple_backtest()
    
    # Comparer plusieurs stratégies
    print("\n=== Comparaison de plusieurs stratégies ===\n")
    results_list, metrics_df = compare_strategies() 