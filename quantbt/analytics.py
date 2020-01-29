import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def performanceSummary(
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
        annVolatility = returns.std() * np.sqrt(260)
    else:
        annVolatility = np.nan

    # Year Fraction (Add a tiny number to prevent divide by zero error)
    yearFraction = (dates[-1] - dates[0]).days / 365 + np.finfo(float).eps

    # Annualized Returns
    annReturns = (nav[-1] / nav[0]) ** (1/yearFraction) - 1

    # Cumulative Returns
    cumReturns = (nav[-1] / nav[0]) - 1

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

    # Downside returns
    downsideReturns = np.array([ret for ret in returns if ret < 0])

    # Downside volatility
    if len(downsideReturns > 1):
        downsideVolatility = downsideReturns.std() * np.sqrt(260)
    else:
        downsideVolatility = np.nan

    # Sortino Ratio
    sortinoRatio = annReturns / (downsideVolatility + np.finfo(float).eps)

    # Calmar Ratio
    calmarRatio = annReturns / (maxdrawdown + np.finfo(float).eps)

    # Latest Cumulative Transaction Costs
    cumulativeTCost = list(historicalTCosts.values())[-1]
    
    # Turnover

    # Gross Leverage

    stats = {
        'Annual Returns'           : annReturns,
        'Annual Volatility'        : annVolatility,
        'Sharpe Ratio'             : annSharpe,
        'Cumulative Return'        : cumReturns,
        'Maximum Drawdown'         : maxdrawdown,
        'Sortino Ratio'            : sortinoRatio,
        'Calmar Ratio'             : calmarRatio,
        'Total Transaction Costs'  : cumulativeTCost
    }

    return stats

def getNAVPlot(port):
    nav = port.getHistoricalNAV()
    name = port.getPortfolioName()
    fig = nav.plot(figsize=(7.5,5),title=name)
    return fig

def getWeightsPlot(port):
    wgt = port.getHistoricalWeights()
    name = port.getPortfolioName()
    fig = wgt.plot(figsize=(7.5,5),title=f'{name} Weights')
    return fig