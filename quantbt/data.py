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

class csvDataHandler:
    def __init__(self,datasources):
        # Initialize a map of file strings to pull csv data from
        self.datasources = datasources
    
    def getDataFromSource(self,source,formatOut='dataframe'):
        sourceName = self.datasources[source]
        try:
            data = pd.read_csv(f'data/{sourceName}',index_col=[0],parse_dates=True)
            dates = list(data.index)
            
            if formatOut.lower() == 'dataframe':
                return [dates,data]
            
            elif formatOut.lower() == 'dictionary':
                return [dates,data.to_dict(orient='index')]

            else:
                raise Exception('ERROR: Invalid Output Format')
        except:
            raise Exception(f'ERROR: Cannot read source {sourceName}')
