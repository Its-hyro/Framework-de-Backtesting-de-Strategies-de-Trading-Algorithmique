import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from ..backtest import BacktestResult


def plot_price_with_signals(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Prix et signaux de trading",
):
    """
    Trace le prix avec les signaux d'achat et de vente.

    Args:
        data: DataFrame contenant les données de prix
        signals: DataFrame contenant les signaux de trading
        figsize: Taille de la figure
        title: Titre du graphique
    """
    plt.figure(figsize=figsize)
    
    # Tracer le prix
    plt.plot(data.index, data["close"], label="Prix", alpha=0.7)
    
    # Tracer les signaux d'achat
    buy_signals = signals[signals["signal"] == 1]
    plt.scatter(
        buy_signals.index,
        data.loc[buy_signals.index, "close"],
        marker="^",
        color="green",
        s=100,
        label="Signal d'achat",
    )
    
    # Tracer les signaux de vente
    sell_signals = signals[signals["signal"] == -1]
    plt.scatter(
        sell_signals.index,
        data.loc[sell_signals.index, "close"],
        marker="v",
        color="red",
        s=100,
        label="Signal de vente",
    )
    
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_strategy_returns(
    result: BacktestResult,
    benchmark: pd.Series = None,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Trace les rendements cumulatifs de la stratégie et du benchmark.

    Args:
        result: Résultat du backtest
        benchmark: Série optionnelle contenant les valeurs d'un benchmark pour comparaison
        figsize: Taille de la figure
    """
    plt.figure(figsize=figsize)
    
    # Calculer les rendements de la stratégie
    strategy_returns = result.portfolio_value.pct_change().dropna()
    strategy_cum_returns = (1 + strategy_returns).cumprod() - 1
    
    # Tracer les rendements cumulatifs de la stratégie
    plt.plot(
        strategy_cum_returns.index,
        strategy_cum_returns,
        label=f"{result.strategy_name} ({result.symbol})",
    )
    
    # Tracer les rendements cumulatifs du benchmark si fourni
    if benchmark is not None:
        benchmark_returns = benchmark.pct_change().dropna()
        benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
        
        plt.plot(
            benchmark_cum_returns.index,
            benchmark_cum_returns,
            label="Benchmark",
            linestyle="--",
        )
    
    plt.title("Rendements cumulatifs")
    plt.xlabel("Date")
    plt.ylabel("Rendement cumulatif")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_underwater(
    result: BacktestResult,
    benchmark: pd.Series = None,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Trace le drawdown (underwater plot) de la stratégie et du benchmark.

    Args:
        result: Résultat du backtest
        benchmark: Série optionnelle contenant les valeurs d'un benchmark pour comparaison
        figsize: Taille de la figure
    """
    plt.figure(figsize=figsize)
    
    # Calculer le drawdown de la stratégie
    strategy_returns = result.portfolio_value.pct_change().dropna()
    strategy_cum_returns = (1 + strategy_returns).cumprod()
    strategy_running_max = strategy_cum_returns.cummax()
    strategy_drawdown = (strategy_cum_returns / strategy_running_max) - 1
    
    # Tracer le drawdown de la stratégie
    plt.fill_between(
        strategy_drawdown.index,
        strategy_drawdown,
        0,
        color="red",
        alpha=0.3,
        label=f"{result.strategy_name} ({result.symbol})",
    )
    plt.plot(strategy_drawdown.index, strategy_drawdown, color="red", alpha=0.5)
    
    # Tracer le drawdown du benchmark si fourni
    if benchmark is not None:
        benchmark_returns = benchmark.pct_change().dropna()
        benchmark_cum_returns = (1 + benchmark_returns).cumprod()
        benchmark_running_max = benchmark_cum_returns.cummax()
        benchmark_drawdown = (benchmark_cum_returns / benchmark_running_max) - 1
        
        plt.fill_between(
            benchmark_drawdown.index,
            benchmark_drawdown,
            0,
            color="blue",
            alpha=0.1,
            label="Benchmark",
        )
        plt.plot(
            benchmark_drawdown.index,
            benchmark_drawdown,
            color="blue",
            alpha=0.5,
            linestyle="--",
        )
    
    plt.title("Drawdown (Underwater Plot)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_rolling_returns(
    result: BacktestResult,
    benchmark: pd.Series = None,
    window: int = 126,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Trace les rendements glissants de la stratégie et du benchmark.

    Args:
        result: Résultat du backtest
        benchmark: Série optionnelle contenant les valeurs d'un benchmark pour comparaison
        window: Taille de la fenêtre glissante (en jours)
        figsize: Taille de la figure
    """
    plt.figure(figsize=figsize)
    
    # Calculer les rendements glissants de la stratégie
    strategy_returns = result.portfolio_value.pct_change().dropna()
    strategy_rolling_returns = strategy_returns.rolling(window=window).mean() * 252
    
    # Tracer les rendements glissants de la stratégie
    plt.plot(
        strategy_rolling_returns.index,
        strategy_rolling_returns,
        label=f"{result.strategy_name} ({result.symbol})",
    )
    
    # Tracer les rendements glissants du benchmark si fourni
    if benchmark is not None:
        benchmark_returns = benchmark.pct_change().dropna()
        benchmark_rolling_returns = benchmark_returns.rolling(window=window).mean() * 252
        
        plt.plot(
            benchmark_rolling_returns.index,
            benchmark_rolling_returns,
            label="Benchmark",
            linestyle="--",
        )
    
    plt.title(f"Rendements glissants annualisés ({window} jours)")
    plt.xlabel("Date")
    plt.ylabel("Rendement annualisé")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_rolling_volatility(
    result: BacktestResult,
    benchmark: pd.Series = None,
    window: int = 126,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Trace la volatilité glissante de la stratégie et du benchmark.

    Args:
        result: Résultat du backtest
        benchmark: Série optionnelle contenant les valeurs d'un benchmark pour comparaison
        window: Taille de la fenêtre glissante (en jours)
        figsize: Taille de la figure
    """
    plt.figure(figsize=figsize)
    
    # Calculer la volatilité glissante de la stratégie
    strategy_returns = result.portfolio_value.pct_change().dropna()
    strategy_rolling_vol = strategy_returns.rolling(window=window).std() * np.sqrt(252)
    
    # Tracer la volatilité glissante de la stratégie
    plt.plot(
        strategy_rolling_vol.index,
        strategy_rolling_vol,
        label=f"{result.strategy_name} ({result.symbol})",
    )
    
    # Tracer la volatilité glissante du benchmark si fourni
    if benchmark is not None:
        benchmark_returns = benchmark.pct_change().dropna()
        benchmark_rolling_vol = benchmark_returns.rolling(window=window).std() * np.sqrt(252)
        
        plt.plot(
            benchmark_rolling_vol.index,
            benchmark_rolling_vol,
            label="Benchmark",
            linestyle="--",
        )
    
    plt.title(f"Volatilité glissante annualisée ({window} jours)")
    plt.xlabel("Date")
    plt.ylabel("Volatilité annualisée")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_returns_distribution(
    result: BacktestResult,
    benchmark: pd.Series = None,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Trace la distribution des rendements de la stratégie et du benchmark.

    Args:
        result: Résultat du backtest
        benchmark: Série optionnelle contenant les valeurs d'un benchmark pour comparaison
        figsize: Taille de la figure
    """
    plt.figure(figsize=figsize)
    
    # Calculer les rendements de la stratégie
    strategy_returns = result.portfolio_value.pct_change().dropna()
    
    # Tracer la distribution des rendements de la stratégie
    sns.histplot(
        strategy_returns,
        kde=True,
        stat="density",
        label=f"{result.strategy_name} ({result.symbol})",
        alpha=0.5,
    )
    
    # Tracer la distribution des rendements du benchmark si fourni
    if benchmark is not None:
        benchmark_returns = benchmark.pct_change().dropna()
        
        sns.histplot(
            benchmark_returns,
            kde=True,
            stat="density",
            label="Benchmark",
            alpha=0.5,
            color="red",
        )
    
    plt.title("Distribution des rendements quotidiens")
    plt.xlabel("Rendement quotidien")
    plt.ylabel("Densité")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_positions(result: BacktestResult, figsize: Tuple[int, int] = (12, 6)):
    """
    Trace l'évolution des positions au fil du temps.

    Args:
        result: Résultat du backtest
        figsize: Taille de la figure
    """
    plt.figure(figsize=figsize)
    
    # Tracer les positions
    plt.plot(result.positions.index, result.positions["position"])
    
    plt.title(f"Positions: {result.strategy_name} sur {result.symbol}")
    plt.xlabel("Date")
    plt.ylabel("Nombre d'unités")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_indicators(
    data: pd.DataFrame,
    indicators: List[str],
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Indicateurs techniques",
):
    """
    Trace les indicateurs techniques sur un graphique.

    Args:
        data: DataFrame contenant les données de prix et les indicateurs
        indicators: Liste des noms des colonnes d'indicateurs à tracer
        figsize: Taille de la figure
        title: Titre du graphique
    """
    plt.figure(figsize=figsize)
    
    # Tracer le prix
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data["close"], label="Prix", alpha=0.7)
    
    # Tracer les indicateurs qui sont sur la même échelle que le prix
    for indicator in indicators:
        if indicator in ["sma_20", "sma_50", "sma_200", "ema_20", "bb_middle", "bb_upper", "bb_lower"]:
            plt.plot(data.index, data[indicator], label=indicator, alpha=0.7)
    
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Tracer les indicateurs qui sont sur une échelle différente
    plt.subplot(2, 1, 2)
    
    for indicator in indicators:
        if indicator not in ["sma_20", "sma_50", "sma_200", "ema_20", "bb_middle", "bb_upper", "bb_lower", "close"]:
            plt.plot(data.index, data[indicator], label=indicator, alpha=0.7)
    
    plt.xlabel("Date")
    plt.ylabel("Valeur")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(
    returns: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Matrice de corrélation des rendements",
):
    """
    Trace la matrice de corrélation des rendements.

    Args:
        returns: DataFrame contenant les rendements de plusieurs actifs
        figsize: Taille de la figure
        title: Titre du graphique
    """
    plt.figure(figsize=figsize)
    
    # Calculer la matrice de corrélation
    corr = returns.corr()
    
    # Tracer la heatmap
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=1,
        cbar=True,
    )
    
    plt.title(title)
    plt.tight_layout()
    plt.show()


def create_performance_tearsheet(
    result: BacktestResult,
    benchmark: pd.Series = None,
    figsize: Tuple[int, int] = (12, 10),
):
    """
    Crée une feuille de performance complète pour une stratégie.

    Args:
        result: Résultat du backtest
        benchmark: Série optionnelle contenant les valeurs d'un benchmark pour comparaison
        figsize: Taille de la figure
    """
    # Calculer les rendements
    strategy_returns = result.portfolio_value.pct_change().dropna()
    
    # Créer la figure
    fig = plt.figure(figsize=figsize)
    
    # Définir la disposition des sous-graphiques
    gs = fig.add_gridspec(3, 2)
    
    # Courbe d'équité
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(result.portfolio_value.index, result.portfolio_value, label="Portfolio")
    if benchmark is not None:
        benchmark_norm = benchmark / benchmark.iloc[0] * result.initial_capital
        ax1.plot(benchmark.index, benchmark_norm, label="Benchmark", alpha=0.7)
    ax1.set_title("Courbe d'équité")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Valeur")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Drawdown
    ax2 = fig.add_subplot(gs[0, 1])
    strategy_cum_returns = (1 + strategy_returns).cumprod()
    strategy_running_max = strategy_cum_returns.cummax()
    strategy_drawdown = (strategy_cum_returns / strategy_running_max) - 1
    ax2.fill_between(strategy_drawdown.index, strategy_drawdown, 0, color="red", alpha=0.3)
    ax2.plot(strategy_drawdown.index, strategy_drawdown, color="red", alpha=0.5)
    ax2.set_title("Drawdown")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)
    
    # Distribution des rendements
    ax3 = fig.add_subplot(gs[1, 0])
    sns.histplot(strategy_returns, kde=True, stat="density", ax=ax3)
    ax3.set_title("Distribution des rendements quotidiens")
    ax3.set_xlabel("Rendement quotidien")
    ax3.set_ylabel("Densité")
    ax3.grid(True, alpha=0.3)
    
    # Rendements mensuels
    ax4 = fig.add_subplot(gs[1, 1])
    monthly_returns = strategy_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
    monthly_returns_df = pd.DataFrame(monthly_returns)
    monthly_returns_df["year"] = monthly_returns_df.index.year
    monthly_returns_df["month"] = monthly_returns_df.index.month
    pivot_table = monthly_returns_df.pivot_table(index="year", columns="month", values=0)
    month_names = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    pivot_table.columns = month_names[:len(pivot_table.columns)]
    sns.heatmap(pivot_table, annot=True, fmt=".2%", cmap="RdYlGn", center=0, linewidths=1, cbar=True, ax=ax4)
    ax4.set_title("Rendements mensuels")
    
    # Positions
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(result.positions.index, result.positions["position"])
    ax5.set_title("Positions")
    ax5.set_xlabel("Date")
    ax5.set_ylabel("Nombre d'unités")
    ax5.grid(True, alpha=0.3)
    
    # Métriques de performance
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")
    
    # Calculer les métriques
    total_return = (result.portfolio_value.iloc[-1] / result.initial_capital) - 1
    annual_return = (1 + total_return) ** (365 / (result.portfolio_value.index[-1] - result.portfolio_value.index[0]).days) - 1
    volatility = strategy_returns.std() * np.sqrt(252)
    sharpe_ratio = (strategy_returns.mean() * 252) / volatility if volatility != 0 else 0
    max_drawdown = strategy_drawdown.min()
    
    # Afficher les métriques
    metrics_text = (
        f"Stratégie: {result.strategy_name}\n"
        f"Symbole: {result.symbol}\n"
        f"Période: {result.start_date} à {result.end_date}\n\n"
        f"Rendement total: {total_return:.2%}\n"
        f"Rendement annualisé: {annual_return:.2%}\n"
        f"Volatilité annualisée: {volatility:.2%}\n"
        f"Ratio de Sharpe: {sharpe_ratio:.2f}\n"
        f"Drawdown maximum: {max_drawdown:.2%}\n"
        f"Nombre de transactions: {len(result.trades)}\n"
    )
    
    ax6.text(0, 1, metrics_text, verticalalignment="top", fontsize=10)
    
    plt.tight_layout()
    plt.show() 