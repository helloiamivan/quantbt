"""Performance metrics and plotting helpers."""

import datetime
from typing import Any, Dict, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


TRADING_DAYS = 260


def _as_python_date(value):
    """Best-effort conversion to ``datetime.date`` for calendar math."""
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    try:
        candidate = value.date()
    except AttributeError:
        return None
    return candidate if isinstance(candidate, datetime.date) else None


def _trading_years_between(dates):
    """Estimate elapsed time in trading years (weekdays / TRADING_DAYS).

    Weekdays are used rather than raw calendar days so that annualisation is
    consistent with the ``TRADING_DAYS`` convention already applied to
    volatility (and to daily fee/borrow accrual in :class:`Portfolio`), while
    still being robust to sparse or weekend-gapped observation dates.
    """
    if len(dates) < 2:
        return 0.0
    d0 = _as_python_date(dates[0])
    d1 = _as_python_date(dates[-1])
    if d0 is None or d1 is None:
        return 0.0
    total_days = (d1 - d0).days
    if total_days <= 0:
        return 0.0
    weekdays = sum(
        1
        for offset in range(total_days)
        if (d0 + datetime.timedelta(days=offset)).weekday() < 5
    )
    return weekdays / TRADING_DAYS


def performanceSummary(
    historicalNAV: Dict[Any, float],
    historicalWeights: Optional[Dict[Any, Dict[str, float]]] = None,
    historicalPositions: Optional[Dict[Any, Dict[str, float]]] = None,
    historicalTCosts: Optional[Dict[Any, float]] = None,
    historicalSlippageCosts: Optional[Dict[Any, float]] = None,
) -> Dict[str, float]:
    """Return annualised performance statistics for a NAV history.

    The unused historical arguments are retained for backwards compatibility.
    Metrics that cannot be calculated from the available observations are NaN.

    Volatility and downside volatility use the sample standard deviation
    (``ddof=1``), matching the convention used in the example risk signals.
    Annualisation of returns uses elapsed weekdays relative to ``TRADING_DAYS``.
    """
    nav = np.asarray(list(historicalNAV.values()), dtype=float)
    dates = list(historicalNAV)
    nan_stats = {
        "Annual Returns": np.nan,
        "Annual Volatility": np.nan,
        "Sharpe Ratio": np.nan,
        "Cumulative Return": np.nan,
        "Maximum Drawdown": np.nan,
        "Sortino Ratio": np.nan,
        "Calmar Ratio": np.nan,
        "Total Transaction Costs": np.nan,
    }
    if not len(nav) or not len(dates) or nav[0] == 0:
        return nan_stats

    returns = np.diff(nav) / nav[:-1]
    annual_volatility = returns.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(returns) > 1 else np.nan
    cumulative_return = nav[-1] / nav[0] - 1
    years = _trading_years_between(dates)
    if years > 0 and 1 + cumulative_return > 0:
        annual_return = (1 + cumulative_return) ** (1 / years) - 1
    else:
        # Not enough history, or the strategy lost its entire capital.
        annual_return = np.nan

    running_max = np.maximum.accumulate(nav)
    maximum_drawdown = np.min(nav / running_max - 1)
    downside_returns = returns[returns < 0]
    downside_volatility = (
        downside_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
        if len(downside_returns) > 1
        else np.nan
    )

    def ratio(numerator, denominator):
        return numerator / denominator if np.isfinite(denominator) and denominator > 0 else np.nan

    transaction_costs = (
        list(historicalTCosts.values())[-1]
        if historicalTCosts
        else 0.0
    )
    return {
        "Annual Returns": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": ratio(annual_return, annual_volatility),
        "Cumulative Return": cumulative_return,
        "Maximum Drawdown": maximum_drawdown,
        "Sortino Ratio": ratio(annual_return, downside_volatility),
        "Calmar Ratio": ratio(annual_return, abs(maximum_drawdown)),
        "Total Transaction Costs": transaction_costs,
    }


def getNAVPlot(port):
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib")
    return port.getHistoricalNAV().plot(figsize=(7.5, 5), title=port.getPortfolioName())


def getWeightsPlot(port):
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib")
    name = port.getPortfolioName()
    return port.getHistoricalWeights().plot(figsize=(7.5, 5), title=f"{name} Weights")


def generateTearsheet(port, save_path: Optional[str] = None):
    """Generate a comprehensive tearsheet/dashboard of backtest results.
    
    Creates a multi-panel figure showing:
    - NAV performance over time
    - Portfolio weights over time
    - Drawdown analysis
    - Performance statistics summary
    
    Args:
        port: Portfolio object with historical data
        save_path: Optional path to save the figure as PNG
        
    Returns:
        matplotlib Figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for tearsheet generation. Install it with: pip install matplotlib")
    
    historical_nav = port.getHistoricalNAV()
    historical_weights = port.getHistoricalWeights()
    historical_cash = port.getHistoricalCash()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"{port.getPortfolioName()} - Backtest Tearsheet", fontsize=16, fontweight='bold')
    
    # NAV plot
    ax1 = plt.subplot(3, 2, 1)
    historical_nav.plot(ax=ax1, title="NAV Performance")
    ax1.set_ylabel("NAV")
    ax1.grid(True, alpha=0.3)
    
    # Weights plot
    ax2 = plt.subplot(3, 2, 2)
    historical_weights.plot(ax=ax2, title="Portfolio Weights")
    ax2.set_ylabel("Weight")
    ax2.grid(True, alpha=0.3)
    
    # Drawdown plot
    ax3 = plt.subplot(3, 2, 3)
    nav_values = historical_nav.iloc[:, 0].values
    running_max = np.maximum.accumulate(nav_values)
    drawdown = (nav_values / running_max - 1) * 100
    ax3.plot(historical_nav.index, drawdown, color='red', label='Drawdown')
    ax3.fill_between(historical_nav.index, drawdown, 0, color='red', alpha=0.3)
    ax3.set_title("Drawdown (%)")
    ax3.set_ylabel("Drawdown %")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Cash plot
    ax4 = plt.subplot(3, 2, 4)
    historical_cash.plot(ax=ax4, title="Cash Account")
    ax4.set_ylabel("Cash")
    ax4.grid(True, alpha=0.3)
    
    # Returns distribution
    ax5 = plt.subplot(3, 2, 5)
    returns = np.diff(nav_values) / nav_values[:-1] * 100
    ax5.hist(returns, bins=30, edgecolor='black', alpha=0.7)
    ax5.set_title("Daily Returns Distribution (%)")
    ax5.set_xlabel("Return %")
    ax5.set_ylabel("Frequency")
    ax5.grid(True, alpha=0.3)
    
    # Performance statistics table
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    stats = port.getPerformanceStatistics(historical=False)
    stats_text = "Performance Statistics:\n\n"
    for col in stats.columns:
        stats_text += f"{col}:\n"
        for idx, val in stats[col].items():
            if isinstance(val, float):
                stats_text += f"  {idx}: {val:.4f}\n"
            else:
                stats_text += f"  {idx}: {val}\n"
        stats_text += "\n"
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
             verticalalignment='top', fontsize=10, family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def exportToExcel(port, file_path: str) -> None:
    """Export backtest results to Excel file.
    
    Creates an Excel file with multiple sheets containing:
    - Historical NAV
    - Historical Weights
    - Historical Positions
    - Historical Cash
    - Performance Statistics
    - Transaction Costs
    
    Args:
        port: Portfolio object with historical data
        file_path: Path to save the Excel file
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for Excel export. Install it with: pip install pandas")
    
    try:
        from openpyxl import Workbook
    except ImportError:
        raise ImportError("openpyxl is required for Excel export. Install it with: pip install openpyxl")
    
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        # Historical NAV
        port.getHistoricalNAV().to_excel(writer, sheet_name='NAV')
        
        # Historical Weights
        port.getHistoricalWeights().to_excel(writer, sheet_name='Weights')
        
        # Historical Positions
        port.getHistoricalPositions().to_excel(writer, sheet_name='Positions')
        
        # Historical Cash
        port.getHistoricalCash().to_excel(writer, sheet_name='Cash')
        
        # Historical Transaction Costs
        port.getHistoricalTCosts().to_excel(writer, sheet_name='Transaction Costs')
        
        # Historical Slippage Costs
        port.getHistoricalSlippageCosts().to_excel(writer, sheet_name='Slippage Costs')
        
        # Historical Borrow Costs
        port.getHistoricalBorrowCosts().to_excel(writer, sheet_name='Borrow Costs')
        
        # Performance Statistics
        port.getPerformanceStatistics(historical=True).to_excel(writer, sheet_name='Performance Stats')
