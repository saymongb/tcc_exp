# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 08:13:16 2019

@author: FlavioFilho
"""
from ML.Arquivo import Arquivo as ar
import matplotlib.pyplot as plt
import pandas as pd


# Data definition, global scope
file = u'Demanda corrediça.xlsx'
path = 'Dataset/'
imagePath = 'Images/'
freq = 'M'
modelo = ['HOLT']
sheet_names = ['2017','2018','2019']
prop=80

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


filiais = pd.read_excel(path+file,'2017')
filiais = filiais['Unidade'].str.replace("  +","") #remover espaços
filiais = filiais.unique()
filiais = filiais[0:28]

for unidade in filiais:
    print(unidade)
    
    serie = readCorredica(unidade)
    
    unidadeAux = ar.removerCaracteresEspeciais(unidade)
    nomePasta = 'ML\\'+unidadeAux
    
    plt.plot(serie)
    plt.title('Corrediça em '+unidade)
    plt.xlabel('Time')
    plt.ylabel('Demand')
    plt.gcf().autofmt_xdate()
    plt.savefig(nomePasta+'\\'+unidadeAux+'- corredica.png',dpi = 300)
    plt.show()
    plt.close()