"""
AlgoTrader - Framework de Backtesting de Stratégies de Trading Algorithmique
"""

__version__ = "0.1.0"
__author__ = "Dorian Drivet"

from . import data
from . import strategy
from . import backtest
from . import portfolio
from . import analysis
from . import visualization

__all__ = ["data", "strategy", "backtest", "portfolio", "analysis", "visualization"]
