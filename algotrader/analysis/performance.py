import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from ..backtest import BacktestResult


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calcule les rendements d'une série de prix.

    Args:
        prices: Série de prix

    Returns:
        Série de rendements
    """
    return prices.pct_change().dropna()


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Calcule les rendements cumulatifs d'une série de rendements.

    Args:
        returns: Série de rendements

    Returns:
        Série de rendements cumulatifs
    """
    return (1 + returns).cumprod() - 1


def calculate_annualized_return(returns: pd.Series) -> float:
    """
    Calcule le rendement annualisé d'une série de rendements.

    Args:
        returns: Série de rendements

    Returns:
        Rendement annualisé
    """
    total_return = (1 + returns).prod() - 1
    n_days = len(returns)
    return (1 + total_return) ** (252 / n_days) - 1


def calculate_volatility(returns: pd.Series) -> float:
    """
    Calcule la volatilité annualisée d'une série de rendements.

    Args:
        returns: Série de rendements

    Returns:
        Volatilité annualisée
    """
    return returns.std() * np.sqrt(252)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calcule le ratio de Sharpe d'une série de rendements.

    Args:
        returns: Série de rendements
        risk_free_rate: Taux sans risque annualisé

    Returns:
        Ratio de Sharpe
    """
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calcule le ratio de Sortino d'une série de rendements.

    Args:
        returns: Série de rendements
        risk_free_rate: Taux sans risque annualisé

    Returns:
        Ratio de Sortino
    """
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    
    if downside_deviation == 0:
        return np.nan
    
    return excess_returns.mean() / downside_deviation * np.sqrt(252)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calcule le drawdown maximum d'une série de rendements.

    Args:
        returns: Série de rendements

    Returns:
        Drawdown maximum
    """
    cumulative_returns = calculate_cumulative_returns(returns)
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / (1 + running_max)
    return drawdown.min()


def calculate_calmar_ratio(returns: pd.Series) -> float:
    """
    Calcule le ratio de Calmar d'une série de rendements.

    Args:
        returns: Série de rendements

    Returns:
        Ratio de Calmar
    """
    annualized_return = calculate_annualized_return(returns)
    max_drawdown = abs(calculate_max_drawdown(returns))
    
    if max_drawdown == 0:
        return np.nan
    
    return annualized_return / max_drawdown


def calculate_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Calcule le ratio Omega d'une série de rendements.

    Args:
        returns: Série de rendements
        threshold: Seuil de rendement

    Returns:
        Ratio Omega
    """
    excess_returns = returns - threshold
    positive_returns = excess_returns[excess_returns > 0].sum()
    negative_returns = abs(excess_returns[excess_returns < 0].sum())
    
    if negative_returns == 0:
        return np.inf
    
    return positive_returns / negative_returns


def calculate_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Calcule la Value at Risk (VaR) d'une série de rendements.

    Args:
        returns: Série de rendements
        alpha: Niveau de confiance (par défaut 5%)

    Returns:
        VaR
    """
    return returns.quantile(alpha)


def calculate_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Calcule la Conditional Value at Risk (CVaR) d'une série de rendements.

    Args:
        returns: Série de rendements
        alpha: Niveau de confiance (par défaut 5%)

    Returns:
        CVaR
    """
    var = calculate_var(returns, alpha)
    return returns[returns <= var].mean()


def calculate_performance_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> Dict[str, float]:
    """
    Calcule diverses métriques de performance pour une série de rendements.

    Args:
        returns: Série de rendements
        risk_free_rate: Taux sans risque annualisé

    Returns:
        Dictionnaire des métriques de performance
    """
    return {
        "total_return": (1 + returns).prod() - 1,
        "annualized_return": calculate_annualized_return(returns),
        "volatility": calculate_volatility(returns),
        "sharpe_ratio": calculate_sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio": calculate_sortino_ratio(returns, risk_free_rate),
        "max_drawdown": calculate_max_drawdown(returns),
        "calmar_ratio": calculate_calmar_ratio(returns),
        "omega_ratio": calculate_omega_ratio(returns),
        "var_5%": calculate_var(returns),
        "cvar_5%": calculate_cvar(returns),
    }


def compare_strategies(results: List[BacktestResult], benchmark: pd.Series = None) -> pd.DataFrame:
    """
    Compare les performances de plusieurs stratégies.

    Args:
        results: Liste des résultats de backtest
        benchmark: Série optionnelle contenant les valeurs d'un benchmark pour comparaison

    Returns:
        DataFrame contenant les métriques de performance pour chaque stratégie
    """
    metrics = []
    
    # Calculer les métriques pour chaque stratégie
    for result in results:
        # Calculer les rendements
        returns = result.portfolio_value.pct_change().dropna()
        
        # Calculer les métriques
        strategy_metrics = calculate_performance_metrics(returns)
        strategy_metrics["strategy_name"] = result.strategy_name
        strategy_metrics["symbol"] = result.symbol
        
        metrics.append(strategy_metrics)
    
    # Calculer les métriques pour le benchmark si fourni
    if benchmark is not None:
        benchmark_returns = benchmark.pct_change().dropna()
        benchmark_metrics = calculate_performance_metrics(benchmark_returns)
        benchmark_metrics["strategy_name"] = "Benchmark"
        benchmark_metrics["symbol"] = "Benchmark"
        
        metrics.append(benchmark_metrics)
    
    # Créer un DataFrame à partir des métriques
    metrics_df = pd.DataFrame(metrics)
    
    # Réorganiser les colonnes
    cols = ["strategy_name", "symbol"]
    cols.extend([col for col in metrics_df.columns if col not in cols])
    metrics_df = metrics_df[cols]
    
    return metrics_df


def plot_equity_curves(results: List[BacktestResult], benchmark: pd.Series = None, figsize: Tuple[int, int] = (12, 6)):
    """
    Trace les courbes d'équité de plusieurs stratégies.

    Args:
        results: Liste des résultats de backtest
        benchmark: Série optionnelle contenant les valeurs d'un benchmark pour comparaison
        figsize: Taille de la figure
    """
    plt.figure(figsize=figsize)
    
    # Tracer la courbe d'équité pour chaque stratégie
    for result in results:
        # Normaliser la courbe d'équité
        equity_curve = result.portfolio_value / result.portfolio_value.iloc[0]
        plt.plot(equity_curve.index, equity_curve, label=f"{result.strategy_name} ({result.symbol})")
    
    # Tracer le benchmark si fourni
    if benchmark is not None:
        # Normaliser le benchmark
        benchmark_norm = benchmark / benchmark.iloc[0]
        plt.plot(benchmark.index, benchmark_norm, label="Benchmark", linestyle="--")
    
    plt.title("Comparaison des courbes d'équité")
    plt.xlabel("Date")
    plt.ylabel("Valeur normalisée")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_drawdowns(results: List[BacktestResult], figsize: Tuple[int, int] = (12, 6)):
    """
    Trace les drawdowns de plusieurs stratégies.

    Args:
        results: Liste des résultats de backtest
        figsize: Taille de la figure
    """
    plt.figure(figsize=figsize)
    
    # Tracer le drawdown pour chaque stratégie
    for result in results:
        # Calculer les rendements
        returns = result.portfolio_value.pct_change().dropna()
        
        # Calculer le drawdown
        cumulative_returns = calculate_cumulative_returns(returns)
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / (1 + running_max)
        
        plt.plot(drawdown.index, drawdown, label=f"{result.strategy_name} ({result.symbol})")
    
    plt.title("Comparaison des drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_monthly_returns_heatmap(returns: pd.Series, figsize: Tuple[int, int] = (12, 8)):
    """
    Trace une heatmap des rendements mensuels.

    Args:
        returns: Série de rendements
        figsize: Taille de la figure
    """
    # Convertir les rendements quotidiens en rendements mensuels
    monthly_returns = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
    
    # Créer un DataFrame avec les rendements mensuels
    monthly_returns_df = pd.DataFrame(monthly_returns)
    monthly_returns_df["year"] = monthly_returns_df.index.year
    monthly_returns_df["month"] = monthly_returns_df.index.month
    
    # Pivoter le DataFrame pour obtenir une matrice années x mois
    pivot_table = monthly_returns_df.pivot_table(
        index="year", columns="month", values=0
    )
    
    # Renommer les colonnes avec les noms des mois
    month_names = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    pivot_table.columns = month_names[:len(pivot_table.columns)]
    
    # Tracer la heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2%",
        cmap="RdYlGn",
        center=0,
        linewidths=1,
        cbar=True,
    )
    
    plt.title("Rendements mensuels")
    plt.tight_layout()
    plt.show()


def plot_rolling_sharpe(returns: pd.Series, window: int = 126, figsize: Tuple[int, int] = (12, 6)):
    """
    Trace le ratio de Sharpe glissant.

    Args:
        returns: Série de rendements
        window: Taille de la fenêtre glissante (en jours)
        figsize: Taille de la figure
    """
    # Calculer le ratio de Sharpe glissant
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
    
    # Tracer le ratio de Sharpe glissant
    plt.figure(figsize=figsize)
    plt.plot(rolling_sharpe.index, rolling_sharpe)
    
    plt.title(f"Ratio de Sharpe glissant ({window} jours)")
    plt.xlabel("Date")
    plt.ylabel("Ratio de Sharpe")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show() 