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

data = [1.0,4,6,5,7,8,5,7,10,20]
index = pd.date_range(start='01/01/2019',periods=10,freq='M')
series = pd.Series(data,index)
start = int(len(series.values)*0.8)
print('Série real:')
print(series)
print()

seletor = ms.ModelSelector(series,1,['CR'],80,stepType='one')
seletor.fit()
'''print('Resultados do seletor:')
print(seletor.modelsResult[0].trainingPrediction)

print(seletor.modelsResult[0].testPrediction)
print(seletor.modelsResult[0].error[0].value)'''

print('Ajuste do AR:')
AR = ar.AR(series[:start])
AR = AR.fit()
trainingFit = pd.Series(AR.fittedvalues)

testPredictions = pd.Series(AR.predict(start=start,end=len(series)-1,dynamic=False))
AR.model.endog = series.values
AR.model.data = series

testPredictions2 = pd.Series(AR.predict(start=start,
                                         end=len(series)-1,
                                         dynamic=False))

print('Treinamento:')
print(trainingFit)
print('Teste')
print(testPredictions)
print('Test2')
print(testPredictions2)
print("Parametros")
print(AR.params)
print("<-------------------------------->")
