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
            # TODO: Enhance the way data is read
            data = pd.read_csv('data/' + sourceName)
            data['Dates'] = pd.to_datetime(data['Dates'])
            data.sort_values(by='Dates',inplace=True)
            dates = data['Dates'].tolist()
            
            if formatOut.lower() == 'dataframe':
                return [dates,data.set_index('Dates')]
            
            elif formatOut.lower() == 'dictionary':
                temp = data.set_index('Dates').T
                return [dates,temp.to_dict()]

            else:
                raise Exception('ERROR: Invalid Output Format')
        except:
            raise Exception(f'ERROR: Cannot read source {sourceName}')
