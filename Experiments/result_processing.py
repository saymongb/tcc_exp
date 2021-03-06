#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:57:00 2019
@author: saymongb
"""
# Imports
import os,sys
sys.path.append(os.getcwd()[0:-11]) # run into tests
import pandas as pd

def improvedScore(dataFrame):
    
    newFrame = pd.Series()
    
    for i in range(len(dataFrame)):
        best = dataFrame.iloc[[i]]['BEST'].values
        best = dataFrame.iloc[[i]][best].values[0]
        newFrame = newFrame.append(pd.Series(best))
       
    return newFrame.mean(),newFrame.std()    

# Data source, directory data
dataFile = 'selection_M3_FULL.xls'
path = '../Results/'
sheetName= ['MMASE','MRMSE']
imagePath = '../Images/'

for sheet in sheetName:
    print()
    print(sheet)
    data = pd.read_excel(path+dataFile,None)
    data = data.pop(sheet)
    mean,std = improvedScore(data)
    print('Mean:'+str(mean)+',std:'+str(std))