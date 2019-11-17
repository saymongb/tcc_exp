import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.initializers import RandomUniform
from keras import backend as K



class ANN(object):
    def __init__(self,dataset,time_delay,horizion, activation_function , stopping_criterion, max_iterations, batch, proportion = 0.80, optimizer = 'adam'):
        print("Entrou")
        
        self.__dataset = np.array(dataset)
        self.__max_iterations = max_iterations
        self.__batch = batch
        self.__horizion = horizion
        self.__time_delay = time_delay
        self.__stopping_criterion = stopping_criterion
        self.__proportion = proportion
        self.__optimizer = optimizer
        self.__activation_function = activation_function
        self.trainScoreMSE = 0
        self.trainScoreRMSE = 0
        self.testScoreMSE = 0
        self.testScoreRMSE = 0
        self.trainPredict = []
        self.testPredict = []
        self.previsoesX = []
        self.previsoesY = []
        self.trainPredictPlot = []
        self.testPredictPlot = []
        self.previsoesPlot = []
        
        # split into train and test sets
        self.__train, self.__test = split_train_and_test(self.__dataset,proportion,time_delay)
        # reshape dataset
        self.__trainX, self.__trainY = create_dataset(self.__train, self.__time_delay)
        self.__testX, self.__testY = create_dataset(self.__test, self.__time_delay)     
        
        
        self.__best_neuron_amount = 1
        self.__best_error = 100000000
        
        #escolhe a quantidade de nerônios na camada escondida
        self.__neuron_optimizer()
        
        # create and fit Multilayer Perceptron model        
        #self.__model = creat_model(time_delay,self.__best_neuron_amount,optimizer,activation_function)
           
    def __neuron_optimizer(self):
        train_aux, test_aux = split_train_and_test(self.__train,self.__proportion,self.__time_delay)
        
        trainX_aux, trainY_aux = create_dataset(train_aux, self.__time_delay)
        testX_aux, testY_aux = create_dataset(test_aux, self.__time_delay) 
        
        for n in range(15):
            nodes = (n+1)
            # create and fit Multilayer Perceptron model
            self.__model = creat_model(self.__time_delay,nodes,self.__optimizer,self.__activation_function)
            # treinando modelo
            if self.__stopping_criterion == 'iterations':
                 self.__model.fit(self.__trainX, self.__trainY, epochs=self.__max_iterations, batch_size=self.__batch, verbose=2)
            else:
                #Vai parar o treinamento da rede quando ela para de melhorar depois de 10 epocas
                es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 10, verbose = 1)
                #Vai diminuir a taxa de aprendizado da rede depois de para de melhorar 5 epocas
                rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)
                #Vai salvar os pesos da melhor epoca de treinamento
                self.__model.fit(trainX_aux, trainY_aux, epochs=self.__max_iterations, batch_size=self.__batch, verbose=2,callbacks = [es,rlr])
                
            print('activation_function:',self.__activation_function)
            print('stopping_criterion:',self.__stopping_criterion)
            print('self.__max_iterations:',self.__max_iterations)
            
            # Estimate model performance
            trainScoreMSE, trainScoreRMSE, testScoreMSE, testScoreRMSE = estimator_Error(self.__model,trainX_aux, trainY_aux,testX_aux, testY_aux)
            
            if self.__best_error > testScoreMSE:
                self.__best_neuron_amount = nodes
                self.__best_error = testScoreMSE
    
    def best_neuron_amount(self):
        return self.__best_neuron_amount, self.__best_error

    def fit(self):
        
          # create and fit Multilayer Perceptron model        
        self.__model = creat_model(self.__time_delay,self.__best_neuron_amount,self.__optimizer,self.__activation_function)
        
        if self.__stopping_criterion == 'iterations':
            self.__model.fit(self.__trainX, self.__trainY, epochs=self.__max_iterations, batch_size=self.__batch, verbose=2)
        else:
            #Vai parar o treinamento da rede quando ela para de melhorar depois de 10 epocas
            es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 10, verbose = 1)
            #Vai diminuir a taxa de aprendizado da rede depois de para de melhorar 5 epocas
            rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)
            #Vai salvar os pesos da melhor epoca de treinamento
            self.__model.fit(self.__trainX, self.__trainY, epochs=self.__max_iterations, batch_size=self.__batch, verbose=2,callbacks = [es,rlr])
            
        print('activation_function:',self.__activation_function)
        print('stopping_criterion:',self.__stopping_criterion)
        print('self.__max_iterations:',self.__max_iterations)
            
        # Estimate model performance
        self.trainScoreMSE, self.trainScoreRMSE, self.testScoreMSE, self.testScoreRMSE = estimator_Error(self.__model,self.__trainX,self.__trainY,self.__testX,self.__testY)
        # generate predictions for training
        self.trainPredict = self.__model.predict(self.__trainX)
        self.testPredict = self.__model.predict(self.__testX)
        #gerar previsões futuras
        self.previsoesX, self.previsoesY = forecating(self.__horizion,self.__time_delay,self.__dataset,self.__model)
        #imprimir resultado
        #self.trainPredictPlot, self.testPredictPlot, self.previsoesPlot = imprimir_Grafico(self.__dataset,self.trainPredict,self.testPredict,self.previsoesY,self.__horizion,self.__time_delay)

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_delay=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_delay):
		a = dataset[i:(i+time_delay)]
		dataX.append(a)
		dataY.append(dataset[i + time_delay])
	return np.array(dataX), np.array(dataY)


def split_train_and_test(dataset,percentage=0.80, time_delay = 1):
    dataset.reshape(( len(dataset),1))
    train_size = int(len(dataset) * percentage)
    train, test = dataset[0:train_size], dataset[(train_size-time_delay):len(dataset)]
    return train, test

def creat_model(time_delay,hidden_nodes,optimization_algorithm,activation_function):
    print('*************')
    print('hidden_nodes: ',hidden_nodes)
    print('*************')
    K.clear_session()
    #incialization = RandomUniform(minval=-0.05, maxval=0.05, seed=76)
    #kernel_initializer=incialization
    model = Sequential()
    if activation_function == 'RELU/LINEAR':
        model.add(Dense(hidden_nodes, input_dim=time_delay, activation='relu'))
        model.add(Dense(1,activation='linear'))
    else:
        model.add(Dense(hidden_nodes, input_dim=time_delay, activation='sigmoid'))
        model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer= optimization_algorithm)
    return model

def estimator_Error(model,trainX,trainY,testX, testY):
    trainScoreMSE = model.evaluate(trainX, trainY, verbose=0)
    trainScoreRMSE = math.sqrt(trainScoreMSE)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScoreMSE, trainScoreRMSE))
    
    testScoreMSE = model.evaluate(testX, testY, verbose=0)
    testScorerMSE = math.sqrt(testScoreMSE)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScoreMSE, testScorerMSE))
    
    return trainScoreMSE, trainScoreRMSE, testScoreMSE, testScorerMSE
    
def forecating(horizion, time_delay,dataset, model):
    previsoesX = np.zeros((horizion, time_delay))
    previsoesY = np.zeros((horizion))
    
    posicaoInicialDosUltimosTres = len(dataset)-(time_delay)
    posicaoFinalDosUltimosTres = len(dataset)
    previsoesX[0] = dataset[posicaoInicialDosUltimosTres:posicaoFinalDosUltimosTres]
    aux = previsoesX[0].reshape((1, time_delay))
    
    for h in range(horizion):
        previsoesY[h] = model.predict(aux)
    
        if h < horizion-1:
            #print(model.predict(aux))
            for t in range(time_delay-1):
                previsoesX[h+1][t] = previsoesX[h][t+1] 
            previsoesX[h+1][time_delay-1] = previsoesY[h]
            aux = previsoesX[h+1].reshape((1, time_delay))
            
    return previsoesX, previsoesY

def imprimir_Grafico(dataset,trainPredict, testPredict, previsoes, horizion,time_delay):
    # shift train predictions for plotting
    trainPredictPlot = np.zeros((len(dataset), 1))
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[time_delay:len(trainPredict)+time_delay, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.zeros((len(dataset), 1))
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(time_delay):len(dataset), :] = testPredict
    #correção da previsão para plotar
    previsoesPlot = np.zeros((horizion+len(dataset), 1))
    previsoesPlot[:, :] = np.nan
    previsoesPlot[(len(dataset)):(horizion+len(dataset))] = previsoes.reshape((horizion, 1))
    # plot baseline and predictions
    plt.plot(dataset)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.plot(previsoesPlot)
    plt.show()
    
    return trainPredictPlot, testPredictPlot, previsoesPlot