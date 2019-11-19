#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 12:02:06 2019

@author: saymongb
"""

# coding=UTF-8
'''

Arquivo gerado para testes com a planilha da corrediça
1. Ler planilha
2. Enviar requisição para o WebService
3. Plotar previsões - em andamento

'''
import os,sys
sys.path.append(os.getcwd()[0:-5]) # run into tests

import matplotlib.pyplot as plt
import pandas as pd
import math
import modelSeletor as ms
import utils.util as ut
import socket

'''print(os.getcwd()[0:-5])
# Data definition, global scope
file = u'Demanda corrediça.xlsx'
path = '..Dataset/'
imagePath = '..Images/'
filial = u'São Paulo'
freq = 'M'
modelo = ['AUTO']
sheet_names = ['2017','2018','2019']
prop=80

print(path+file)

def readCorredica(filial):
    #pasta de trabalho
    temp = pd.Series()
    
    for sheet in sheet_names:
        
        spr = pd.read_excel(path+file,sheet)
        spr['Unidade'] = spr['Unidade'].str.replace("  +","") #remover espaços
        newSpr = spr[spr['Unidade'] == filial] # apenas da filial especificada
        timeSeries = pd.Series(data=newSpr['Quantidade'].values,
                          index=newSpr['Data'].values)
        temp = temp.append(timeSeries)
        
    sample =  temp.resample(freq)
    timeSeries = sample.sum()    
    
    return timeSeries

def buildNewSerie(testValues,forecast,minDate):
    
    dateIndexTest = pd.date_range(start=minDate,
                                  periods=len(testValues),
                                  freq=freq)
    testSeries = pd.Series(testValues,dateIndexTest)
    
    dateIndexForecast = pd.date_range(start=dateIndexTest[-1],
                                      periods=len(forecast),
                                      freq=freq)
    forecastSeries = pd.Series(forecast,dateIndexForecast)
    return testSeries,forecastSeries

filiais = pd.read_excel(path+file,'2017')
filiais = filiais['Unidade'].str.replace("  +","") #remover espaços
filiais = filiais.unique()
print(filiais)

filiais = ['São Paulo']#,'POA - Central SUL','Recife','Rio de Janeiro']

for unidade in filiais:
 
    serie = readCorredica(unidade)
    teste = serie.astype('double')
    serie = pd.Series(teste)
    md = ms.ModelSelector(serie,6,modelo)
    md.fit()
    
    start = math.floor(len(serie)*(prop/100))
    testPredictions, forecasts = buildNewSerie(md.modelsResult[0].testPrediction.to_numpy(),
                     md.modelsResult[0].forecastDemand.to_numpy(),
                     serie.index[start])
    
    print()
    print(unidade)
    print()
    modeloTemp = md.modelsResult[0].model
    plt.plot(serie,'black',testPredictions,'blue',forecasts,'green')
    plt.title('Corrediça em '+unidade)
    plt.xlabel('Time')
    plt.ylabel('Demand')
    plt.legend(['Real','Test', modeloTemp])
    plt.gcf().autofmt_xdate()
    plt.savefig(imagePath+unidade+'- corredica.png',dpi = 300)
    plt.close()'''