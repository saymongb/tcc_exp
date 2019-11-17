import modelSeletor as ms
import pandas as pd
import json

#timeseries = pd.Series(t)
#print(timeseries)


t = []
for x in range(10):
    t.append(x*1.0)
    
t = [1,2.0, 2.0,0,0,0,3,4,0,0,0,2]
model = ms.ModelSelector(t,10,['NAIVE','SES'])
model.fit()
print(t)
print("Treianamento:",model.trainingPrediction[0])
print("Teste:",model.testPrediction[0])
print("Previsão:",model.forecasts[0])
print("Erro:",model.testError[0])
print()

print(t)
print("Treianamento:",model.trainingPrediction[1])
print("Teste:",model.testPrediction[1])
print("Previsão:",model.forecasts[1])
print("Erro:",model.testError[1])
print()

p =model.forecasts[1]
teste = {"demand": p.tolist()}
print(teste)
print(model.testError[1])
print(model.testError[0])


