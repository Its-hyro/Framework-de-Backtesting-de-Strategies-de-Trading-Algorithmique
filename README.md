# AlgoTrader - Framework de Backtesting de Stratégies de Trading Algorithmique

## Description
AlgoTrader est un framework Python conçu pour le backtesting de stratégies de trading algorithmique. Il permet aux traders et aux analystes quantitatifs de développer, tester et optimiser des stratégies de trading sur des données historiques avant de les déployer sur les marchés réels.

## Installation

```bash
# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

```python
from algotrader.data import YahooFinanceDataLoader
from algotrader.strategy import MovingAverageCrossover
from algotrader.backtest import Backtester
from algotrader.portfolio import Portfolio

# Charger les données
data_loader = YahooFinanceDataLoader()
data = data_loader.load("AAPL", start_date="2020-01-01", end_date="2021-01-01")

# Créer une stratégie
strategy = MovingAverageCrossover(short_window=20, long_window=50)

# Initialiser le portfolio
portfolio = Portfolio(initial_capital=10000)

# Exécuter le backtest
backtester = Backtester(data, strategy, portfolio)
results = backtester.run()

# Analyser les résultats
results.plot_equity_curve()
print(f"Sharpe Ratio: {results.sharpe_ratio}")
print(f"Max Drawdown: {results.max_drawdown}%")
```

## Compilation 
```bash
source venv_py310/bin/activate && cd examples && python simple_backtest.py
```
