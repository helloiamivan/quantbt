"""Performance metrics and plotting helpers."""

import datetime

import numpy as np


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
    historicalNAV,
    historicalWeights=None,
    historicalPositions=None,
    historicalTCosts=None,
    historicalSlippageCosts=None,
):
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
    return port.getHistoricalNAV().plot(figsize=(7.5, 5), title=port.getPortfolioName())


def getWeightsPlot(port):
    name = port.getPortfolioName()
    return port.getHistoricalWeights().plot(figsize=(7.5, 5), title=f"{name} Weights")
