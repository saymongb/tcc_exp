from objects import StudyResults
import modelSeletor as ms
import utils.util as ut

class ForecastStudy(object):
    def __init__(self):
        print()
        
    #Método para fazer executar novo treinamento e predição de modelos
    #models é uma lista dos modelos que que deseja treinar
    #demands é uma lista das demandas que serão usadas no treinamento
    #forecast_Horizon é quantos périodos para frente os modelos iram fazer previsão
    #part é o nome da peça que se quer prever a demanda
    #place é o lugar da peça  que se quer prever a demanda
    def addForecastingStudy(self, models,demands,forecast_Horizon,part,remoteStation,frequency):
        
        #Chamado uma função para validar alguns parametros
        #A função retornar um status de verificação se está tudo certo
        #Ela também retorna mensagens de erro apontando o parâmentro errado
        studyStauts, msg = self.validaForecastingStudy(models,demands,forecast_Horizon)
        if(studyStauts):
            
            listResult = []
            jsonListModels = []
                  
            #Instância a classe de treinamento de modelos e executa o treinamento
            demandSeries = ut.Utils.listDictToPandas(demands,frequency)
            model = ms.ModelSelector(demandSeries,
                                     forecast_Horizon,
                                     models)
            
            demandSeriesStauts, msg = self.validateAmountOfObservationOfDemands(demandSeries)
            
            if(demandSeriesStauts):
            
                model = ms.ModelSelector(demandSeries,
                                         forecast_Horizon,
                                         models)
                
                model.fit()
                listResult = model.getModelResults()
    
                   
                for m in listResult:
                    
                    m.part = part
                    m.remoteStation = remoteStation
                    jsonListModels.append(m.json())
                
                
                listStudyResults = StudyResults(demandSeries,jsonListModels)
                
                return True, listStudyResults.json()
            return False, msg
        return False, msg
    
    def validaForecastingStudy(self, models,demands,forecast_Horizon):
        #Valida se forecast_Horizon realmente é um número
        try:
            
            Horizon = 2 + forecast_Horizon

        except Exception:
            return False, "Error decoding forecast_Horizon, check your json."

        #Valida se demands realmente é uma lista de número
        try:
            a = demands[0]
            demand = float(a['demandValue']) 
            
        except Exception:
            return False, "Error decoding demands, check your json."
            
        #Valida se os modelos de treinamento passados realmente existem
        for m in list(models):
            if(not m in ms.ModelSelector.getModelsNamesList()):
                return False, "Name invalid to models."
        
        return True, "Ok"
    
    def validateAmountOfObservationOfDemands(self,demandSeries):
        minimum_amount_of_demand = 12
        msg = ''
        status = True
        if len(demandSeries) < minimum_amount_of_demand:
                msg = 'Error, the amount of demand is insufficient, the minimum amount of 12 observations'
                status = False
        return status, msg
            
