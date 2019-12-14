#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 10:20:27 2019

@author: saymongb
"""

import os,sys
sys.path.append(os.getcwd()[0:-11])
import statsmodels.tsa.holtwinters as ts
import statsmodels.tsa.ar_model as ar
import pandas as pd
import matplotlib.pyplot as plt
import modelSeletor as ms

data = [1.0,4,2,6,7,8,5,11,10,20]
index = pd.date_range(start='01/01/2019',periods=10,freq='M')
series = pd.Series(data,index)
start = int(len(series.values)*0.8)
print('Série real:')
print(series)
print()
# Fit a model
SES = ts.Holt(series[:start])
SES = SES.fit(optimized = True,use_brute = True)
SESParams = SES.params
SES2  = ts.Holt(series)
SES2 = SES2.fit(smoothing_level=SESParams['smoothing_level'],
              optimized=False,
              smoothing_slope = SESParams['smoothing_slope'],
                        initial_slope = SESParams['initial_slope'],          
              initial_level=SESParams['initial_level'],
              use_brute=False)

print('Treinamento:')
print(SES.fittedvalues)
print()
print('Teste')
print(SES2.fittedvalues)
print()
seletor = ms.ModelSelector(series,1,['SES'],80,stepType='multi')
seletor.fit()
print('Resultados do seletor:')
print(seletor.modelsResult[0].trainingPrediction)

print(seletor.modelsResult[0].testPrediction)
print(seletor.modelsResult[0].error[0].value)

