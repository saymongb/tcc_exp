#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 22:30:13 2019
Author:     Saymon G. Bandeira
Contact:    saymongb@gmail.com
Summary:    Code to run model selection experiment and save results.
Note:       a) Save results in a ".xlsx" file.
            b) Tests with all models in pool.

Fix: 
Status: Validating         
Next: Generate data without selection of combination (equal)
       
"""

# Imports
import os,sys
sys.path.append(os.getcwd()[0:-11]) # run into tests
import matplotlib.pyplot as plt
import pandas as pd
import objects as obj
import utils.util as ut
import numpy as np
import datetime as dt
import math
import modelSeletor as ms
import warnings
import re # REGEX Operation

warnings.filterwarnings("ignore")

def improvedScore(dataFrame,benchmark):
    
    numOfImproved = 0 
    
    for i in range(len(dataFrame)):
        
        best = dataFrame.iloc[[i]]['BEST'].values
        best = dataFrame.iloc[[i]][best].values
        mean = dataFrame.iloc[[i]][benchmark].values[0]
       
        if best < mean:
            numOfImproved +=1
       
    return (numOfImproved/(len(dataFrame)))*100    

# Data source, directory data
#dataFile = u'Demanda corrediça.xlsx'
dataFile = 'M3C.xls'
path = '../Dataset/'
imagePath = '../Images/Error/'
resultsPath = '../Results/'
outputFileName = None
m3 = dataFile=='M3C.xls'

# Experiment configuration
metrics= ['MASE','RMSE','MAPE']
horizon = 1
frequency = ['M']#,'W','D']
modelsList = ['NAIVE','SES','HOLT','AR','CR','CF1']
proportionList = [60,20,20]
combinationType= ['errorBased','equal']

colors = np.random.rand(len(modelsList),3) # to plot
data = pd.read_excel(path+dataFile,None)

if not m3:
    
    # Specific corrediça spreadsheet operations
    outputFileName = 'selection_corredica'
    sheet_names = ['2017','2018','2019'] # For corrediça spreadsheet
    filiais = pd.read_excel(path+dataFile,'2017')
    filiais = filiais['Unidade'].str.replace("  +","") #remove space
    names = filiais.unique()
    names.sort()
    #names = ['Porto Velho','São Paulo'] # coment this line for executions
    
else:
    
    # Specific to M3-Competition monthly data
    # Extract series conform to https://doi.org/10.1016/j.jbusres.2015.03.028
    outputFileName = 'selection_M3_30'
    frequency = ['M']
    data = data.pop('M3Month')
    #data = data[data['N']>=60]
    names = data['Series'].unique()
    names.sort()
    #names = ['N2801','N1404','N1417','N1793','N1428'] # coment this line for executions

# To compute time of executions
startTime = dt.datetime.now()
totalTime = None

# Excel frame structure, changes os this data may affects buildSpreadSheet's
# behavior on Utils class.
cols = ['Series Name','SES','HOLT','NAIVE',
        'AR','CR','CF-Mean','CF-Error','BEST',
        'Model','Mean','Std.','% Improved']
frame = pd.DataFrame(columns = cols)
# To save on disk
writer = pd.ExcelWriter(resultsPath+outputFileName+'.xls')

for metric in metrics:
    print(metric)
    for freq in frequency:
        
        frame = pd.DataFrame(columns = cols)
        
        for serieName in names:
            
            print()
            print('Series:'+serieName)
        
            if not m3:
                # Get data
                serie = ut.Utils.readCorredica(serieName,data,sheet_names)
                # Proccess series on same frequency
                sampleObj = serie.resample(freq)
                newSeries  = sampleObj.sum()
            else:
                newSeries = ut.Utils.buildM3DataFrame(data,serieName)
                
            # Set intervals for evaluation
            # 1--------T1--------T2--------->T
            T1 = int(len(newSeries)*(proportionList[0]/100))
            T2 = int(len(newSeries)*((proportionList[0]+proportionList[1])/100))
            
            # Fit model on validation data and get best
            validation = ms.ModelSelector(data=newSeries[:T2],
                                        models=modelsList,
                                        start=T1,
                                        combType=combinationType[0],
                                        combMetric=metric)
            
            test = ms.ModelSelector(data=newSeries,
                                        models=modelsList,
                                        start=T2,
                                        combType=combinationType[0],
                                        combMetric=metric)
            
            validation.fit()
            test.fit()
            
            combinationByError = validation.getModelByName('CF-Error')
            validation.removeModel('CF-Error')
            validation.combType = 'equal'
            validation.combinationFit()   
            validation.modelsResult.append(combinationByError)
            
            bestValidation,value = validation.getBestByMetric(metric)
            
            combinationByError = test.getModelByName('CF-Error')
            test.removeModel('CF-Error')
            test.combType = 'equal'
            test.combinationFit()   
            test.modelsResult.append(combinationByError)
            
            
            # Add to DataFrame
            line = {}
            for m in test.modelsResult:
                
                errorValue = obj.ForecastErro.getValueByMetricName(m.error,metric)
                line[m.model] = errorValue
                
            line['Series Name'] = serieName
            line['BEST'] = bestValidation.model
             
            frame = frame.append(line,ignore_index=True)
        
        improvedPct = improvedScore(frame,'NAIVE')
        frame = frame.append({'% Improved':round(improvedPct,4)},ignore_index=True)
        
        legends = []
        for m in frame.columns[1:8]:
            
            modelName = m
            meanError = frame[m].mean()
            std = frame[m].std()
            frame = frame.append({'Model':modelName,
                                                'Mean':meanError,
                                                'Std.':std},
                                              ignore_index=True)
            
        frame.to_excel(excel_writer=writer,sheet_name=freq+metric,index=False)
    
writer.save()

totalTime = dt.datetime.now() - startTime
print()
print('Time to run tests:')
print(totalTime)
print()