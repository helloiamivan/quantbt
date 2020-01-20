import pandas as pd
import numpy as np

def summaryStatistics(
    historicalNAV,
    historicalWeights,
    historicalPositions,
    historicalTCosts,
    historicalSlippageCosts):

    # Get price levels as list
    nav = list(historicalNAV.values())

    # Get dates as list
    dates = list(historicalNAV.keys())

    # Daily Percentage Returns vector
    returns = np.diff(nav) / nav[:-1]

    # Annualized Volatility
    if len(returns) > 1:
        annVolatility = returns.std() * np.sqrt(252)
    else:
        annVolatility = np.nan

    # Year Fraction (Add a tiny number to prevent divide by zero error)
    yearFraction = (dates[-1] - dates[0]).days / 365 + np.finfo(float).eps

    # Annualized Returns
    annReturns = (nav[-1] / nav[0]) ** (1/yearFraction) - 1

    # Annualized Sharpe Ratio (Add a tiny number to prevent divide by zero error)
    annSharpe = annReturns / ( annVolatility + np.finfo(float).eps )

    # End of Max Drawdown period
    maxdrawdownEnd = np.argmax(np.maximum.accumulate(nav) - nav)

    # Start of Max Drawdown period
    if maxdrawdownEnd != 0:
        maxdrawdownStart = np.argmax(nav[:maxdrawdownEnd])
    else:
        maxdrawdownStart = 0

    # Maximum Drawdown
    maxdrawdown = nav[maxdrawdownEnd] / nav[maxdrawdownStart] - 1

    stats = {
        'Ann. Returns'       : annReturns,
        'Ann. Volatility'    : annVolatility,
        'Ann. Sharpe Ratio'  : annSharpe,
        'Maximum Drawdown'   : maxdrawdown
    }

    return stats