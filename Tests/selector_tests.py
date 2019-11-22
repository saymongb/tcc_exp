# coding=UTF-8
'''
Arquivo gerado para testes com a planilha da corrediça
1. Ler planilha
2. Enviar requisição para o WebService
3. Plotar previsões

Fix: change read's on excel file to same as experiment
'''
import os,sys
sys.path.append(os.getcwd()[0:-6]) # run into tests

import matplotlib.pyplot as plt
import pandas as pd
import requests
import json
import objects as obj
import utils.util as ut
import numpy as np
import croston as cr

# Croston tests.

# Example from OTexts
'''data = [0,2,0,1,0,11,0,0,0,0,2,0,6,3,0,0,0,0,0,7,0,0,0,0,0,0,0,3,1,0,0,1,0,1,0,0]
index = pd.date_range(start='01/01/2019',periods=len(data),freq = 'M')
data = pd.Series(data,index)
md = cr.Croston(data=data,init='naive',alpha=0.1)
md.fit()
print(md.forecast(2))

#data2.index=index
#plt.plot(md.data)
#plt.plot(data2)
#plt.gcf().autofmt_xdate()
'''
# Data definition, global scope
file = u'Demanda corrediça.xlsx'
path = '../Dataset/'
imagePath = '../Images/'
resultsPath = '../Results/'
frequency = 'M'#2W, 3W,D,W
modelo = ['AR','SES']#['AR','HOLT','SES','NAIVE','CF1','CR']
sheet_names = ['2017','2018','2019']
# ip da máquina virtual/Windows 7
#url = "http://localhost:8090/forecastingMethods/Statistical/"
url = 'http://ec2-18-189-180-102.us-east-2.compute.amazonaws.com:8090/forecastingMethods/Statistical'
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
prop=80
cores = np.random.rand(len(modelo),3)

filiais = pd.read_excel(path+file,'2017')
filiais = filiais['Unidade'].str.replace("  +","") #remover espaços
filiais = filiais.unique() 
filiais = ['Brasília','Salvador','Rio Branco','São Paulo']
filiais.sort()

# for excel file
cols = ['Unidade','AR','HOLT','SES','NAIVE','CF1','Min Value']
resultSheet = pd.DataFrame(columns=cols)

dados = pd.read_excel(path+file,None)

for unidade in filiais:
 
    serie = ut.Utils.readCorredica(unidade,dados,sheet_names)
    
    data = {'models': modelo,
            'frequency':frequency,
            'demands': obj.Demand.getJSONDemandList(serie),
            'horizon': 6,
            'part': 'Corrediça',
            'remoteStation': unidade
    }
    
    # Send request
    r = requests.post(url, data=json.dumps(data), headers=headers)
    studyJson=r.json()
    
    print('Processando unidade:'+unidade)
    
    # Proccess series on same frequency
    sampleObj = serie.resample(frequency)
    newSeries  = sampleObj.sum()
    
    demands = ut.Utils.listDictToPandas(studyJson['processedDemand'],frequency)
    testPreds = ut.Utils.listDictToPandas(studyJson['modelsResults'][0]['testPrediction'],frequency)
        
    #plt.plot(demands,c='black')
    finalModelList = ['Real']
    rmseValues = []
    
    '''for i in range(len(studyJson['modelsResults'])):
        
        rmseValues.append(studyJson['modelsResults'][i]['error'][0]['value'])
        #rmseValues.append(studyJson[i]['error'][0]['value'])
        
        nome = studyJson['modelsResults'][i]['model']
        testPreds = ut.Utils.listDictToPandas(studyJson['modelsResults'][i]['testPrediction'],frequency)
        forecasts = ut.Utils.listDictToPandas(studyJson['modelsResults'][i]['forecastDemand'],frequency)
        forecasts = testPreds.append(forecasts)
        finalModelList.append(nome)
        #finalModelList.append('Forecast of:'+nome)
        #plt.plot(testPreds,c=cores)
        #plt.plot(forecasts,c=cores[i][:])
    
    rmseValues = pd.array(rmseValues)
    
    line = {'Unidade':unidade,
            'AR':rmseValues[0],
            'HOLT':rmseValues[1],
            'SES':rmseValues[2],
            'NAIVE':rmseValues[3],
            'CF1':rmseValues[4],
            'Min Value':rmseValues.min()}
    resultSheet = resultSheet.append(line,ignore_index=True)'''
    
    #demands = demands[len(demands)-len(testPreds):]
    plt.plot(newSeries,'black')
    plt.plot(testPreds,'r')
    plt.title('Corrediça em '+unidade)
    plt.xlabel('Time')
    plt.ylabel('Demand')
    plt.legend(finalModelList)
    plt.gcf().autofmt_xdate()
    plt.savefig(imagePath+unidade+'- corredica.png',dpi = 800)
    plt.close()
    
    # Build plot
    #plt.plot(newSeries,'black',
    #         ut.Utils.listDictToPandas(studyJson[0]['testPrediction'],frequency),'blue',
    #         ut.Utils.listDictToPandas(studyJson[0]['forecastDemand'],frequency),'green')
    #plt.title('Corrediça em '+unidade)
    #plt.xlabel('Time')
    #plt.ylabel('Demand')
    #plt.legend(['Real','Test', studyJson[0]['model']])
    #plt.gcf().autofmt_xdate()
    #plt.savefig(imagePath+unidade+'- corredica.png',dpi = 300)
    #plt.close()
#resultSheet.to_excel(excel_writer=resultsPath+'Metrics.xlsx',index=False)