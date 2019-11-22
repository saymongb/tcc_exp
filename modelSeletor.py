'''
Created on 24 de jun de 2019
Author:     Saymon G. Bandeira
Contact:    saymongb@gmail.com
Summary:    Framework to perform model selection and get evaluation metrics
Note:       Implement first item level selection, include ARIMA models.
            Works with Pandas.Series
            Detail parameters data!
            Models in pool: Simple Exponential Smoothing (SES), 
                            Naive Forecast (NAIVE),
                            Autoregressive (AR), 
                            SES Holt modification (HOLT),
                            Croston Method (CR),
                            Combination Forecast (CF),
                            Automatic selection procedure based on RMSE (AUTO)
Fix: Method for removal a model.
Status: Implement Croston methods. Implement trimmed combination.        
'''

import statsmodels.tsa.holtwinters as ts
import statsmodels.tools.eval_measures as ms
import statsmodels.tsa.ar_model as ar
import math
import objects as obj
import utils.util as ut
import numpy as np
import pandas as pd
import sys
import croston as cr

class ModelSelector:
    
    # Class variables
    _metrics = ['RMSE','MAPE','MASE']
    _standardModels = ['SES','NAIVE','AR','HOLT','CR','CF1']
    _horizonLimit = 12
    _decimalPlaces= 4
    
    def __init__(self,data,horizon=1,models=['AUTO'], prop=80,start=None,
                 combType='equal', combMetric='RMSE'):
        '''Parameters
           ----------
           
            data: a Pandas.Series representing a historic of an item (time-series)
            horizon: number of forecats required
            start: index to initial forecast on training set
            prop: proportion of training/test-sample to fit a model, default 80-20.
            combType: type of combination ['equal','errorBased','trimmed'] method to be used
            combMetric: metric used to compute the weights of each method
        '''
        # set train and test data
        if start:
            self.start = start
        else:    
            self.start = math.ceil(len(data)*(prop/100))
        
        self.data = data
        self.trainData = data[:self.start]
        self.testData = data[self.start:]
         
        self.modelsName = models
        self.fittedModel = None # used as auxiliary variable
        self.fittedModelFinal = None # used as auxiliary variable
        self.horizon = horizon
        self.prop = prop
        self.numModels = len(models)
        self.noCount = 0 # number of models not disregarded to combination
        self.combType = combType
        self.combMetric = combMetric
        
        self.modelsResult = []
        
        self.initialize()
     
    def initialize(self):
        '''
            Purpose: adjust parameters and set fields to avoid internal errors.
        '''
        # 1. "AUTO" and "CF" models must be the first and last on modelsName
        # 2. Set number of models to be disregarded.
        idx_last = self.modelsName.count('AUTO')
        idx_second_to_last = self.modelsName.count('CF1')
        
        if idx_second_to_last > 0:
            
            self.modelsName.remove('CF1')
            self.modelsName.append('CF1') #last
            self.noCount = 1
        
        if idx_last > 0:
        
            self.modelsName.remove('AUTO')
            self.modelsName.insert(0,'AUTO') #first
            self.noCount = self.noCount=1
            self.numModels = len(ModelSelector._standardModels)
        
        self.numModels = self.numModels - self.noCount
        
    def fit(self):
        '''Fit each mode and:
            a) set training/test predictions
            b) set model error(training/test) values
            c) set forecasts
        '''
        for i in range(len(self.modelsName)):

            if not self.isFitted(self.modelsName[i]):
            
                if self.modelsName[i] in ['SES','NAIVE','HOLT']:
                   
                    self.exponentialFit(self.modelsName[i])
                    
                elif self.modelsName[i] == 'CR':
                    
                    self.crostonFit()
                    
                elif self.modelsName[i] == 'AR':
                    
                    self.ARFit()
                    
                elif self.modelsName[i] == 'AUTO':
                    
                    self.autoFit()
                    
                elif self.modelsName[i] == 'CF1':
                    
                    self.combinationFit()
                
    def exponentialFit(self,name):
        ''' Parameters
            ----------
            
            name: name of model
        '''    
        modelName = name
        errorObjs = []
        
        # Step 1: fit selected model
        if name == 'NAIVE':
             # for evaluation
             self.fittedModel = ts.ExponentialSmoothing(self.trainData)
             self.fittedModel = self.fittedModel.fit(smoothing_level=1)
             # for real forecasts
             self.fittedModelFinal = ts.ExponentialSmoothing(self.data)
             self.fittedModelFinal = self.fittedModelFinal.fit(smoothing_level=1)
             
        elif name == 'SES':
            # for evaluation
             self.fittedModel = ts.SimpleExpSmoothing(self.trainData)
             self.fittedModel = self.fittedModel.fit(optimized = True,
                                                   use_brute = True) #grid search
            # for real forecasts
             self.fittedModelFinal = ts.SimpleExpSmoothing(self.data)
             self.fittedModelFinal = self.fittedModelFinal.fit(optimized = True,
                                                  use_brute = True) #grid search
        elif name == 'HOLT':
            # Holt-Winters 
            # for evaluation
             self.fittedModel = ts.Holt(self.trainData)
             self.fittedModel = self.fittedModel.fit(optimized = True,
                                                   use_brute = True) #grid search
            # for real forecasts
             self.fittedModelFinal = ts.Holt(self.data)
             self.fittedModelFinal = self.fittedModelFinal.fit(optimized = True,
                                                  use_brute = True) #grid search
            
        # Step 2: get fitted values for training, test and forecasts
        trainingFit = pd.Series(self.fittedModel.fittedvalues)
        testPredictions = pd.Series(self.fittedModel.forecast(len(self.testData)))
        forecasts = pd.Series(self.fittedModelFinal.forecast(self.horizon))
               
        # Step 3: set error
        errorObjs = self.setErrorData(trainingFit,testPredictions)
        
        # Add to ModelsResult list
        self.setModelResults(modelName,errorObjs,trainingFit,
                            testPredictions,forecasts)
    def crostonFit(self):
        
        modelName = 'CR'
        errorObjs = []
        
        # Step 1: fit selected model
        
        self.fittedModel = cr.Croston(self.trainData)
        self.fittedModel.fit()
        
        self.fittedModelFinal = cr.Croston(self.data)
        self.fittedModelFinal.fit()
        
        # Step 2: get fitted values for training, test and forecasts
        trainingFit = pd.Series(self.fittedModel.fittedForecasts)
        testPredictions = pd.Series(self.fittedModel.forecast(len(self.testData)))
        forecasts = pd.Series(self.fittedModelFinal.forecast(self.horizon))
               
        # Step 3: set error
        errorObjs = self.setErrorData(trainingFit,testPredictions)
        
        # Add to ModelsResult list
        self.setModelResults(modelName,errorObjs,trainingFit,
                            testPredictions,forecasts)
        
    def ARFit(self):
        ''' Fits a autoregressive model.
        '''    
        modelName = 'AR'
        errorObjs = []
        
        # Step 1: set training and test values
        self.fittedModel = ar.AR(self.trainData)
        self.fittedModel = self.fittedModel.fit()
        trainingFit = pd.Series(self.fittedModel.fittedvalues)
        testPredictions = pd.Series(self.fittedModel.predict(
                start=len(self.trainData),
                end=len(self.trainData)+len(self.testData)-1,
                dynamic=False))
        
        # Step 2: Training again with all data for acurate forecasts
        self.fittedModelFinal = ar.AR(self.data)
        self.fittedModelFinal = self.fittedModelFinal.fit()
        forecasts = pd.Series(self.fittedModelFinal.predict(
                start=len(self.data),
                end=len(self.data)+self.horizon-1,
                dynamic=False))
        
        '''Step 3: set error
            for AR, the size of trainData will be different from
            fitted values at model. Fill initial trainingPredictions
            with same data as real. This will no affect the evaluation
            metrics.
        '''
        initialValues = self.data[:len(self.trainData)-len(trainingFit)]
        trainingFit = initialValues.append(trainingFit)
        errorObjs = self.setErrorData(trainingFit,testPredictions)
        
        # Add to ModelsResult list
        self.setModelResults(modelName,errorObjs,trainingFit,
                            testPredictions,forecasts)
        
    def fitAllModels(self):
        '''
            The verification in isFitted is necessary due to the order on which
            the models are passed (random).
        '''
        
        if not self.isFitted('SES'):
          
            self.exponentialFit('SES')
        
        if not self.isFitted('NAIVE'):
            
            self.exponentialFit('NAIVE')
        
        if not self.isFitted('HOLT'):
            
            self.exponentialFit('HOLT')
       
        if not self.isFitted('CR'):        
                
            self.crostonFit()
        
        if not self.isFitted('AR'):        
                
            self.ARFit()
        
        if not self.isFitted('CF1'):
            
            self.combinationFit()
        
    def autoFit(self):
        ''' Automatic selection based on minimal RMSE.
            
            Parameters:
            -----------
            fitType: {"best","all"}, retain best model or all.
        '''
       
        self.fitAllModels()
        
        minError = sys.float_info.max
        finalModel = []
        
        for m in self.modelsResult:
            
            if m.error[0].value < minError:
                minError = m.error[0].value
                finalModel = m
        
        self.modelsResult.clear()
        self.modelsResult.append(finalModel)
    
    def getBestByMetric(self,metricName):
        
        minError = sys.float_info.max
        finalModel = []
        
        for m in self.modelsResult:
            
            errorValue = obj.ForecastErro.getValueByMetricName(m.error,metricName)
            
            if errorValue < minError:
                    minError = errorValue
                    finalModel = m
                    
        return finalModel,minError            
                    
    def getModelByName(self,name):
        
        for m in self.modelsResult:
            if m.model == name:
                return m
        return 'Invalid'    
                
     
    def combinationFit(self):
        '''
            Linear combination of models. Some matrix algebra is necessary here. 
            Using NumPy arrays for calculations.
            
            Variables
            ---------
            
            coefs: a 1xN vector of weights of each model in pool. The weights
                can be equal for all models or assigned conform to some criteria.
                
            traningMatrix,testMatrix,forecastMatrix: Each column is a point on
                time and each row, the forecasts made by one model. The 
                combinations are performed over the columns.
        '''
        
        # Step 1: set weights
        coefs = self.setWeights()
        # Step 2: build matrix of training/test/forecast values
        traningMatrix = np.array([])
        testMatrix = np.array([])
        forecastMatrix = np.array([])
        
        sizeTraining = len(self.trainData)
        sizeTest = len(self.testData)
        
        # Same indexes for all models, including combination.
        trainingIdx = self.modelsResult[0].trainingPrediction.index
        testIdx = self.modelsResult[0].testPrediction.index
        forecastIdx = self.modelsResult[0].forecastDemand.index
        
        for m in self.modelsResult:
            
            if m.model not in ['CF-Mean','CF-Error']:
                traningMatrix = np.append(traningMatrix,
                                      m.trainingPrediction.values)
                testMatrix = np.append(testMatrix,
                                   m.testPrediction.values)
                forecastMatrix = np.append(forecastMatrix,
                                       m.forecastDemand.values)
        
        # Reshape to get matrix
        traningMatrix = np.reshape(traningMatrix,[self.numModels, sizeTraining])
        testMatrix = np.reshape(testMatrix,[self.numModels, sizeTest])
        forecastMatrix = np.reshape(forecastMatrix,[self.numModels, self.horizon])
        
        # Step 3: compute forecasts by matrix multiplication
        if self.combType == 'equal':
            modelName = 'CF-Mean'
        else:
            modelName = 'CF-Error'
        
        if self.combType == 'trimmed':
            # Exclude min and max from each predictions (lines)
            trainingFit = ModelSelector.makeTrimmedMean(traningMatrix)
            testPredictions = ModelSelector.makeTrimmedMean(testMatrix)
            forecasts = ModelSelector.makeTrimmedMean(forecastMatrix)
            
        else: 
        
            trainingFit = np.matmul(coefs,traningMatrix)
            testPredictions = np.matmul(coefs,testMatrix)
            forecasts = np.matmul(coefs,forecastMatrix)
        
        trainingFit = pd.Series(trainingFit,trainingIdx)
        testPredictions = pd.Series(testPredictions,testIdx)
        forecasts = pd.Series(forecasts,forecastIdx)
           
        
        errorObjs = self.setErrorData(trainingFit,testPredictions)
        
        # Add to ModelsResult list        
        self.setModelResults(modelName,errorObjs,trainingFit,
                            testPredictions,forecasts)
        
       
    def isFitted(self,modelName):
        
        for m in self.modelsResult:
            
            if m.model == modelName:
                return True
        return False
 
    @staticmethod
    def makeTrimmedMean(predictionsMatrix):
        '''
            Removes the upper and lower output of the predictors and
            returns a mean of remaining values.
            
            Actually working removing 2 predictions (one upper, one lower)
            
            Parameter
            ---------
            predictionsMatrix: a mxn numpy matrix.
                        
        '''
        # Step 1: sort columns
        predictionsMatrix = np.sort(predictionsMatrix,axis=0)
        
        # Step 2: remove first and last rows
        predictionsMatrix = np.delete(predictionsMatrix,
                                      [0, len(predictionsMatrix)-1],0)
        
        # Step 3: return the mean over lines
        final = np.mean(predictionsMatrix,0)
        
        return final
    
    def setWeights(self):
        
        coefVector = np.zeros(self.numModels)
        metricValues = []
        sum = 0
        
        #combType,combMetric
        
        if self.combType == 'equal':
        
            coefVector [:] = 1/self.numModels
            coefVector = np.array(coefVector)
            
        elif self.combType == 'errorBased':
            
            coefVector = []
            
            for m in self.modelsResult:
                
                val = obj.ForecastErro.getValueByMetricName(m.error,self.combMetric)
                metricValues.append(val)
                sum = sum + val
            
            for m in metricValues:
                
                coefVector.append(1-(m/sum))    
            
            coefVector = np.array(coefVector)
            coefVector[:] = coefVector[:]/coefVector.sum() 

        return coefVector
        
    # Build a ForecastErro list
    def setErrorData(self,trainingFit,testPredictions):
        
        auxList = []
        
        # Root Mean Squared Error - RMSE
        trainingErrorRMSE = round(ms.rmse(self.trainData,trainingFit),
                                  ModelSelector._decimalPlaces)
        testErrorRMSE = round(ms.rmse(self.testData,testPredictions),
                              ModelSelector._decimalPlaces)
        auxList.append(obj.ForecastErro('RMSE',testErrorRMSE,'TEST'))
        auxList.append(obj.ForecastErro('RMSE',trainingErrorRMSE,'TRAIN'))
        
        #MAPE only all values > 0
        if 0 not in self.data.values:
             
            trainingErrorMAPE = round(ut.Utils.mape(self.trainData,trainingFit),
                                      ModelSelector._decimalPlaces)
            testErrorMape = round(ut.Utils.mape(self.testData,testPredictions),
                                  ModelSelector._decimalPlaces)
            auxList.append(
                    obj.ForecastErro('MAPE',trainingErrorMAPE,'TRAIN'))
            auxList.append(
                    obj.ForecastErro('MAPE',testErrorMape,'TEST'))        
        
        # Mean Absolute Scaled Error
        trainingErrorMASE = round(ut.Utils.mase(self.trainData.to_numpy(),
                                                self.trainData.to_numpy(),
                                                trainingFit.to_numpy()),
                                      ModelSelector._decimalPlaces)
        testErrorMASE = round(ut.Utils.mase(self.trainData.to_numpy(),
                                            self.testData.to_numpy(),
                                            testPredictions.to_numpy()),
                                  ModelSelector._decimalPlaces)
        auxList.append(
                obj.ForecastErro('MASE',trainingErrorMASE,'TRAIN'))
        auxList.append(
                obj.ForecastErro('MASE',testErrorMASE,'TEST')) 
        
        return auxList
    
    # Build ModelResult object
    def setModelResults(self,model,error,trainingPrediction,
                       testPrediction,forecastDemand):
        
        # Replace negative values
        trainingPrediction[trainingPrediction<0] = 0
        testPrediction[testPrediction<0] = 0
        forecastDemand[forecastDemand<0] = 0
        
        self.modelsResult.append(
                obj.ModelsResult(model,error,trainingPrediction,
                       testPrediction,forecastDemand)
                )
                
     # Getters,Setters and support functions.
    def getModelResults(self):
        return self.modelsResult
     
    def removeModel(self,name):
        
        model = self.getModelByName(name)
        self.modelsResult.remove(model)
    
    @staticmethod
    def getModelsNamesList():
        # AUTO must be appended here due the recursion in autofit method.
        a = ModelSelector._standardModels.copy()
        a.append('AUTO')
        return a
    
    @staticmethod
    def getMetricsNames():
        return ModelSelector._metrics     

    @staticmethod
    def getHorizonLimit():
        return ModelSelector._horizonLimit