import os
import sys
import time
import json
import math
import copy
import base64
import getpass
import warnings
import requests
import logging

import numpy as np
import pandas as pd
from datetime import datetime

from .analytics import summaryStatistics, getNAVPlot

def flattenDictionary(nestedDict):
    listofDict = []

    for key,value in nestedDict.items():
        flatDict = nestedDict[key]
        flatDict['Dates'] = key
        listofDict.append(flatDict.copy())

    return listofDict

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print(f'Error: Creating directory {directory}')

# Core Portfolio Object
class Portfolio:
    def __init__(self,positions,cash,name='',datadump=False,backtestFolderName=os.getcwd()):
        
        # Portfolio Parameters
        self.name = name
        self.positions = positions
        self.cash = cash
        self.datadump = datadump
        self.fixedTransactionCosts = dict()
        self.annualManagementFee = 0.0
        self.slippageModel = ''
        self.impactParams = dict()
        self.unwindUndefinedAssetWeights = True
        self.transactionCosts = 0.0
        self.slippageCosts = 0.0
        self.LastRebalanceDate = 'N/A'
        self.FirstRebalanceDate = 'N/A'
        self.performanceStatistics = dict()

        # Custom Data to serialize
        self.customData = dict()

        # Time Series Data
        self.historicalNAV = dict()
        self.historicalPositions = dict()
        self.historicalWeights = dict()
        self.historicalTCosts = dict()
        self.historicalSlippageCosts = dict()

        # Utils        
        self.timestamp = ''.join(str(time.time()).split('.'))
        self.backtestFolderName = backtestFolderName + '/BackTestResults/' + self.timestamp + '-' + self.name

        # Create folder to store backtest results
        createFolder(self.backtestFolderName)

    # Get Methods
    def getPortfolioName(self):
        return self.name
    
    def getBacktestFolderName(self):
        return self.backtestFolderName

    def getPositions(self):
        return self.positions

    def getAssetPosition(self,asset):
        return self.positions[asset]
    
    def getCash(self):
        return self.cash

    def getTransactionCosts(self):
        return self.transactionCosts
    
    def getAssetsInPortfolio(self):
        return list(self.positions.keys())

    def getFixedTransactionCosts(self,asset):
        return self.fixedTransactionCosts[asset]
    
    def getAllFixedTransactionCosts(self):
        return self.fixedTransactionCosts

    def getAnnualManagementFee(self):
        return self.annualManagementFee
    
    def getBacktestTimestamp(self):
        return self.timestamp

    def getFirstRebalanceDate(self):
        return self.FirstRebalanceDate
    
    def getLastRebalanceDate(self):
        return self.LastRebalanceDate

    def getSlippageModel(self):
        return self.slippageModel

    def getSlippageCosts(self):
        return self.slippageCosts
    
    def getImpactParams(self):
        return self.impactParams

    def getCustomData(self):
        return self.customData

    def getCustomDataByDate(self,date):
        if date in list(self.customData.keys()):
            return self.customData[date]
        else:
            return dict()
    
    def getWeights(self,lastPriceMap):
        weights = {}
        currentNAV = self.getNAV(lastPriceMap)

        for asset in self.getPositions():
            price = lastPriceMap[asset]
            weights[asset] = price * self.getAssetPosition(asset) / currentNAV
        
        return weights
    
    def getHistoricalWeights(self,formatOut='DataFrame'):
        if formatOut.lower() == 'dataframe':
            flatDictionary = flattenDictionary(self.historicalWeights)
            temp = pd.DataFrame.from_dict(flatDictionary).set_index('Dates')
            return temp
        
        elif formatOut.lower() == 'dictionary':
            return self.historicalWeights
        
        else:
            raise Exception('ERROR: Invalid Historical Weight Output Format')

    def getNAV(self,lastPriceMap):
        positionValue = float(0.0)

        # Mark to market all assets
        for asset in self.getPositions():
            positionValue += lastPriceMap[asset] * self.getAssetPosition(asset)
        
        # Add cash account
        value = positionValue + self.getCash()

        return value

    def getHistoricalNAV(self,formatOut='DataFrame'):
        if formatOut.lower() == 'dataframe':
            temp = pd.DataFrame(self.historicalNAV,index=[0],).T
            temp.columns = ['Historical NAV']
            return temp
        elif formatOut.lower() == 'dictionary':
            return self.historicalNAV
        else:
            raise Exception('ERROR: Invalid Historical NAV Output Format')

    def getHistoricalPositions(self,formatOut='DataFrame'):
        if formatOut.lower() == 'dataframe':
            flatDictionary = flattenDictionary(self.historicalPositions)
            temp = pd.DataFrame.from_dict(flatDictionary).set_index('Dates')
            return temp
        
        elif formatOut.lower() == 'dictionary':
            return self.historicalPositions
        
        else:
            raise Exception('ERROR: Invalid Historical Positions Output Format')
    
    def getHistoricalTCosts(self,formatOut='DataFrame'):
        if formatOut.lower() == 'dataframe':
            temp = pd.DataFrame(self.historicalTCosts,index=[0],).T
            temp.columns = ['Cumulative Transaction Costs']
            return temp
        elif formatOut.lower() == 'dictionary':
            return self.historicalTCosts
        else:
            raise Exception('ERROR: Invalid Historical Transaction Costs Output Format')

    def getHistoricalSlippageCosts(self,formatOut='DataFrame'):
        if formatOut.lower() == 'dataframe':
            temp = pd.DataFrame(self.historicalSlippageCosts,index=[0],).T
            temp.columns = ['Cumulative Slippage Costs']
            return temp
        elif formatOut.lower() == 'dictionary':
            return self.historicalSlippageCosts
        else:
            raise Exception('ERROR: Invalid Historical Slippage Costs Output Format')
    
    def getPerformanceStatistics(self,historical=False):
        perfStats = pd.DataFrame.from_dict(self.performanceStatistics,orient='index')
        
        if historical == False:
            return perfStats.iloc[[-1]]
        else:
            return perfStats
    
    # Set Methods
    def setCash(self,cash):
        if isinstance(cash,float):
            self.cash = cash
        else:
            raise Exception('ERROR: Cash must be a float number')

    def setFixedTransactionCosts(self,fixedTransactionCosts):
        if isinstance(fixedTransactionCosts,dict):
            self.fixedTransactionCosts = fixedTransactionCosts
        else:
            raise Exception('ERROR: Transaction costs must be a dictionary of asset names and costs')
    
    def setSlippageModel(self,slippageModel):
        validSlippageModels = ['squarerootimpact']

        if slippageModel in validSlippageModels:
            self.slippageModel = slippageModel
        else:
            raise Exception(f'ERROR: Choose from {",".join(validSlippageModels)}')
    
    def setImpactParams(self,impactParams):
        if isinstance(impactParams,dict):
            self.impactParams = impactParams
        else:
            raise Exception('ERROR: Impact Parameters must be a dictionary')
    
    def setAnnualManagementFee(self,annualManagementFee):
        self.annualManagementFee = annualManagementFee

    def setFirstRebalanceDate(self,date):
        if isinstance(date,pd.Timestamp):
            self.FirstRebalanceDate = date
        else:
            raise Exception('ERROR: First Rebalance Date must be a pandas datetime object')
    
    def setLastRebalanceDate(self,date):
        if isinstance(date,pd.Timestamp):
            self.LastRebalanceDate = date
        else:
            raise Exception('ERROR: Last Rebalance Date must be a pandas datetime object')

    def setCustomData(self,date,data_dict):
        if isinstance(date,pd.Timestamp) and isinstance(data_dict,dict):
            self.customData[date] = data_dict
        else:
            raise Exception('ERROR: Custom data must be a dictionary with pandas datetime as keys')
    
    def setUnwindUndefinedAssetWeights(self,unwindUndefinedAssetWeights):
        self.unwindUndefinedAssetWeights = unwindUndefinedAssetWeights

    def setTransactionCosts(self,transactionCosts):
        self.transactionCosts = transactionCosts

    def setSlippageCosts(self,slippageCosts):
        self.slippageCosts = slippageCosts
    
    # Analytics Functions
    def plotNAV(self):
        return getNAVPlot(self)

    # Portfolio Object Methods
    def buy(self,asset,quantity,lastPriceMap):
        if asset in self.getAssetsInPortfolio():
            self.positions[asset] += quantity
            if self.getAssetPosition(asset) < 0:
                # Adjust cash account if asset still has an overall short position
                self.setCash(self.getCash() + (-1 * self.getAssetPosition(asset) * lastPriceMap[asset]))
        else:
            self.positions[asset] = quantity
    
    def sell(self,asset,quantity,lastPriceMap):
        if asset in self.getAssetsInPortfolio():
            self.positions[asset] -= quantity
            if self.getAssetPosition(asset) < 0:
                # Adjust cash account if asset still has an overall short position
                self.setCash(self.getCash() + (-1 * self.getAssetPosition(asset) * lastPriceMap[asset]))
        else:
            self.positions[asset] = -1 * quantity
            self.setCash(self.getCash() + (quantity * lastPriceMap[asset]))
    
    def signOff(self,date,lastPriceMap):
        # Add annual management fees
        managementFee = self.getNAV(lastPriceMap) * self.getAnnualManagementFee() * (1/252)
        
        # TODO: Add interest in cash account
        # interestCash = self.getCash() * overnightLIBOR * (1/252)

        self.setCash(self.getCash() - managementFee)

        # Historical daily states
        self.historicalPositions[date] = copy.deepcopy(self.getPositions())
        self.historicalNAV[date] = float(copy.deepcopy(self.getNAV(lastPriceMap)))
        self.historicalWeights[date] = copy.deepcopy(self.getWeights(lastPriceMap))
        self.historicalTCosts[date] = float(copy.deepcopy(self.getTransactionCosts()))
        self.historicalSlippageCosts[date] = float(copy.deepcopy(self.getSlippageCosts()))

        # Compute Performance Statistics
        self.performanceStatistics[date] = summaryStatistics(
            self.historicalNAV,
            self.historicalWeights,
            self.historicalPositions,
            self.historicalTCosts,
            self.historicalSlippageCosts)

        # Serialize Data
        if self.datadump == True:
            dataDump = self.getBacktestFolderName() + '/' + date.strftime('%Y-%m-%d') + '.json'

            dailyNode = [{
                'Date'                 : date.strftime('%Y-%m-%d'),
                'NAV'                  : self.historicalNAV[date],
                'Cash'                 : self.getCash(),
                'FirstRebalanceDate'   : self.getFirstRebalanceDate(),
                'LastRebalanceDate'    : self.getLastRebalanceDate(),
                'TransactionCosts'     : self.getTransactionCosts(),
                'FixedTransactionCosts': self.getAllFixedTransactionCosts(),
                'SlippageModel'        : self.getSlippageModel(),
                'Positions'            : self.getPositions(),
                'Weights'              : self.getWeights(lastPriceMap),
                'CustomData'           : self.getCustomDataByDate(date),
                'Performance'          : self.performanceStatistics[date]
            }]

            # TODO: Fix date serialization instead of string defaults
            with open(dataDump,'w') as fd:
                fd.write(json.dumps(dailyNode, indent=2, default=str))
    
    # Default rebalance function
    def rebalance(self,targetWeights,lastPriceMap,date):
        # Current NAV is the market to market using latest positions, current close prices and cash account
        currentNAV = self.getNAV(lastPriceMap)

        # Clean up the cash account at every rebalance
        # This assumes we buy lesser units when transaction costs are factored in rather than having residual cash balances
        self.setCash(0.0)

        # Initialize Slippage Costs
        slippageCosts = 0.0

        # Initialized Fixed Transaction Costs
        tCosts = 0.0

        # Save the most recent rebalance date
        self.LastRebalanceDate = date

        # Initialize the first rebalance date
        if self.getFirstRebalanceDate() == 'N/A':
            self.setFirstRebalanceDate(date)
        
        # Define the treatment of undefined assets which are previously in the portfolio
        if self.unwindUndefinedAssetWeights == True:
            for asset in self.getAssetsInPortfolio():
                # Unwind assets not in target weights dictionary
                if (asset in list(targetWeights.keys())) == False:

                    if self.getAssetPosition(asset) > 0:
                        self.sell(asset,self.getAssetPosition(asset),lastPriceMap[asset])
                    else:
                        self.buy(asset,-1*self.getAssetPosition(asset),lastPriceMap[asset])

                    if len(self.fixedTransactionCosts) > 0:
                        tCosts = abs(self.getAssetPosition(asset) * lastPriceMap[asset] * self.getFixedTransactionCosts(asset))
                    
                    if self.getSlippageModel() == 'squarerootimpact':
                        impactParams = self.getImpactParams()
                        ADV = impactParams[asset]['ADV']
                        vol = impactParams[asset]['Volatility']
                        spreadCost = impactParams[asset]['BidAskSpread']
                        scalingFactor = impactParams[asset]['ScalingFactor']

                        slippage = spreadCost + scalingFactor*(1/math.sqrt(252))*vol*math.sqrt(abs(self.getAssetPosition(asset))/ADV)

                        slippageCosts = abs(self.getAssetPosition(asset)) * lastPriceMap[asset] * slippage

                        # Account for slippage in cash account
                        self.setCash(self.getCash() - slippageCosts)

                        # Add to cumulative transaction costs
                        self.setSlippageCosts(self.getSlippageCosts + slippageCosts)

        # Loop through all target trade intentions
        for asset in targetWeights:
            singleAssetTargetUnits = targetWeights[asset] * currentNAV / lastPriceMap[asset]

            if asset in self.getPositions():
                unitsToTrade = singleAssetTargetUnits - self.getAssetPosition(asset)
            else:
                unitsToTrade = singleAssetTargetUnits

            # Note: Sell/Buy function only accept POSITIVE quantities, the sign is handled in the methods
            if unitsToTrade >= 0:
                self.buy(asset,unitsToTrade,lastPriceMap)
            
            else:
                self.sell(asset,-1*unitsToTrade,lastPriceMap)

            # Account for fixed transaction costs
            if len(self.fixedTransactionCosts) > 0:

                tCosts = abs(unitsToTrade) * lastPriceMap[asset] * self.getFixedTransactionCosts(asset)

                # Deduct transaction costs from cash account
                self.setCash(self.getCash() - tCosts)

                # Add to cumulative transaction costs
                self.setTransactionCosts(self.getTransactionCosts() + tCosts)

            # Account for slippage costs
            if self.getSlippageModel() == 'squarerootimpact':
                impactParams = self.getImpactParams()
                ADV = impactParams[asset]['ADV']
                vol = impactParams[asset]['Volatility']
                spreadCost = impactParams[asset]['BidAskSpread']
                scalingFactor = impactParams[asset]['ScalingFactor']

                # Do not account for the impact of funding portfolio in slippage
                if date == self.getFirstRebalanceDate():
                    slippage = 0.0
                else:
                    slippage = spreadCost + scalingFactor*(1/math.sqrt(252))*vol*math.sqrt(abs(self.getAssetPosition(asset))/ADV)
                
                slippageCosts = abs(unitsToTrade) * lastPriceMap[asset] * slippage

                # Deduct slippage from cash account
                self.setCash(self.getCash() - slippageCosts)

                # Add to cumulative slippage costs
                self.setSlippageCosts(self.getSlippageCosts() + slippageCosts)

        # Account for leverage in the portfolio
        # Check long positions and how much cash we need to borrow against
        longPositions = sum([targetWeights[asset] if targetWeights[asset] > 0 else 0 for asset in targetWeights])

        excessCash = (1.0 - longPositions) * currentNAV

        # Adjust cash account
        self.setCash(self.getCash() + excessCash)






            



        

        