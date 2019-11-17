from flask_restplus import fields

#Esse arquivo Python possui as classes importantes para manipular na Api

# Class to represent a single demand data
class Demand(object):
    
    def __init__(self,value,date):
        '''Params: value, a float value
                   date, a string representation of a date
        '''
        self.demandValue = value
        self.demandDate = date
        
    def json(self):
        return {
                "demandValue":self.demandValue,
                "demandDate":self.demandDate
                }
        
    @staticmethod
    def getJSONDemandList(demandList):
        # This method iterate over a Pandas.Series object
        
        temp = []
        values = demandList.values
        indexes = demandList.index.strftime('%d/%m/%Y')
        
        for value,index in zip(values,indexes):
            temp.append(Demand(float(value),index).json())
        return temp
    

#Classe para manipular os erros de métrica dos modelos
class ForecastErro(object):
    def __init__(self, name, value,errorType='TEST'):
         #TEST = Erro do conjunto de teste, caso contrário treinamento (TRAIN)
        self.name = name
        self.value = value
        self.errorType = errorType
    #Função para criar json de retorno
    def json(self):
        return {
                "name":self.name,
                "value":self.value,
                "errorType":self.errorType
                }
   
    @staticmethod
    def getJSONerroList(listError):
       temp = []
       for m in listError:
           temp.append(m.json())
       return temp    
   
    # Fixed to TEST set
    @staticmethod    
    def getValueByMetricName(errorList,metricName):
        
        for er in errorList:
            
            if er.errorType == 'TEST' and er.name == metricName:
                return er.value 
    

#Classe para manipular os resultados dos modelos
class ModelsResult(object):
    
    def __init__(self, model,error,trainingPrediction,testPrediction,
                 forecastDemand,part=None,remoteStation=None):
        self.model = model
        self.part = part
        self.remoteStation = remoteStation
        self.error = error
        self.trainingPrediction = trainingPrediction
        self.testPrediction = testPrediction
        self.forecastDemand = forecastDemand
    
    #Função para criar json de retorno   
    def json(self): 

        return {
                "model":self.model,
                "part":self.part,
                "remoteStation":self.remoteStation,
                "trainingPrediction": Demand.getJSONDemandList(self.trainingPrediction),
                "testPrediction": Demand.getJSONDemandList(self.testPrediction), 
                "forecastDemand": Demand.getJSONDemandList(self.forecastDemand),
                "error": ForecastErro.getJSONerroList(self.error)
                }
        
class StudyResults(object):
    def __init__(self,processedDemand,modelsResults):
        self.processedDemand = processedDemand
        self.modelsResults = modelsResults
        
    def json(self):
            
        return{"processedDemand":Demand.getJSONDemandList(self.processedDemand),
                    "modelsResults":self.modelsResults
                    }
  
#Classe para manipular as documentações dos modelos de reposta e entrada de json
class documentationModel(object):
    
    def __init__(self,app):
        
        self.demandData = app.model('DemandData',
                                    {'demandValue': fields.Integer(required=True,
                                                                   description = 'The value of a single demand.',
                                                                   example=12),
                                     'demandDate': fields.String(required=True,
                                                                 description = 'A string representation of a date in the format DD/MM/YYYY.',
                                                                 example='01/12/2019')})
        

    
        
        self.modelForecastingStudy = app.model(
                'ForecastingStudy',
				  {'models': fields.List(fields.String(enum=['NAIVE','SES','AR','HOLT','CR','AUTO','CF1']),required = True, description="Names of the forecasting methods", example=['NAIVE','SES','AR','HOLT','CR','AUTO','CF1']), 
                    'frequency': fields.String(enum=['M','W','D','A'],required = True, description="The frequency of series (M=Month,W=Week,D=Day and A=Year).", example='M'),
                    'demands': fields.List(fields.Nested(self.demandData),description="The historical (time-series) demand data.",required=True),
                    'horizon': fields.Integer(required = True,description="Number of periods to forecast", min=1, example=1),
                    'part':fields.String(description="Part name", example='Bico Dosador'),
                    'remoteStation':fields.String(description="Demand location", example='São Paulo'),},)

        self.modelErroMetric = app.model('ForecastErro', 
                                         {'name':fields.String(enum=['RMSE','MAPE','MASE'],required = True,description="Name of the  forecating erro",example='RMSE'),
                                          'value': fields.Float(required = True,description="Value of the  forecating erro",example=15.5),
                                          'errorType': fields.String(enum=['TEST','TRAIN'],required = True,description="The set name where metrics were calculated.",example='TRAIN')},)
    
        self.modelsResult = app.model('ModelsResult', 
				  {'model': fields.String(enum=['NAIVE','SES','AR','HOLT','CR','CF1'],required = True, description="Name of the forecasting method"),
                   'trainingPrediction': fields.List(fields.Nested(self.demandData),description="Predicted demands in trainning set."),
                    'testPrediction': fields.List(fields.Nested(self.demandData),description="Predicted demands in test set."),
                    'forecastDemand': fields.List(fields.Nested(self.demandData),description="Predicted demands for the horizon parameter."),
                    'error': fields.List(fields.Nested(self.modelErroMetric),description="Error of forecating"),
                    'part':fields.String(required = True,description="Part name", example='Bico Dosador'),
                    'remoteStation':fields.String(required = True,description="Demand location", example='São Paulo'),},)
        
        self.modelStudyResults = app.model('StudyResults', 
                                         {'processedDemand': fields.List(fields.Nested(self.demandData),description="Processed demand conform to specified granuarity."),
                                          'modelsResults': fields.List(fields.Nested(self.modelsResult),description="List of model results")})
    
        
        
    def getmodelForecastingStudy(self):
        return self.modelForecastingStudy
    
    def getmodelsResult(self):
        return self.modelsResult
    
    def getmodelStudyResults(self):
        return self.modelStudyResults
    
#Classe personaliza de exeception
class ErrorJson(Exception):
   "Error decoding json, check your json."
   pass
    
