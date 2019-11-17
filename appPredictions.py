from flask import Flask, request
from flask_restplus import Api, Resource, reqparse, inputs
from objects import documentationModel,ErrorJson
from study import ForecastStudy


#from waitress import serve

#instanciando a Api Flask
flask_app = Flask(__name__)

#Configurando parâmetros do Flask
flask_app.config.setdefault('RESTPLUS_MASK_SWAGGER', False)

flask_app.config['PREFERRED_URL_SCHEME'] = 'http'
flask_app.config["SERVER_NAME"] = "localhost:8090"
#flask_app.config["SERVER_NAME"] = "ec2-18-189-180-102.us-east-2.compute.amazonaws.com:8090"
#flask_app.config.SWAGGER_VALIDATOR_URL = 'http://domain.com/validator'

#instanciando documentação
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "Predictions  API", 
		  description = "Have algorithm for demand forecasting")

#,base_url='/test',doc='/documentation', prefix="/v0.1"

#Configurando documentação do namespace para as chamdas de métodos
name_space = app.namespace('forecastingMethods', description='provides several methods for forecasting time series')

#Instanciando classe que cria a documentação dos modelos de json
documentation = documentationModel(app)
modelForecastingStudy = documentation.getmodelForecastingStudy()
modelsResult = documentation.getmodelsResult()
modelStudyResults = documentation.getmodelStudyResults()

parser = reqparse.RequestParser()
parser.add_argument('url', type=inputs.URL(schemes=['http']))
#Criando uma rota para chamada de métodos
@name_space.route("/Statistical/")
class MainClass(Resource):
    
    parser = reqparse.RequestParser()
    parser.add_argument('url', type=inputs.URL(schemes=['http']))
    
    #definindo o modelo de retorno do método
    @app.marshal_with(modelStudyResults)
    #definindo os códigos de reposta
    @app.doc(responses={ 200: 'OK', 400: 'Invalid Argument', 500: 'Internal Server Error' })
    #definindo o modelo de entrada do método
    @app.expect(modelForecastingStudy)	
    
    
    #definindo um método post para entrada de dados
    def post(self):
        try:
            
            #capturando os dados recebidos pelo json
            try:
                demands = request.json['demands']
                
            except Exception:
                raise ErrorJson("Error decoding demands, check your json.")
            try:
                models = request.json['models']
                
            except Exception:
                raise ErrorJson("Error decoding models, check your json.")  
            try:
               forecast_Horizon = request.json['horizon']
            except Exception:
                raise ErrorJson("Error decoding horizon, check your json.")
            try:
                part = request.json['part']
            except Exception:
                raise ErrorJson("Error decoding part, check your json.")
            try:
                remoteStation = request.json['remoteStation']
            except Exception:
                raise ErrorJson("Error decoding remoteStation, check your json.")
            try:
                frequency = request.json['frequency']
            except Exception:
                raise ErrorJson('Error decoding frequency, check your json')
                
            
            #Instanciando a classe que valida, executa o treinamento e a predição
            F = ForecastStudy()
            status, res = F.addForecastingStudy(models,
                                                demands,
                                                forecast_Horizon,
                                                part,
                                                remoteStation,
                                                frequency)
            
            if status:
                return res
            else:
               raise ErrorJson(res)

       
        except ErrorJson as e:
            print(e.args[0])
            name_space.abort(400, e.args[0], status = "Invalid Argument", statusCode = "400")
            
        except Exception as e:
            print(e.args[0])
            name_space.abort(500, e.args[0], status = "Internal Server Error", statusCode = "500")
            

#Defini a execução da Api
#if __name__ == '__main__':
    #serve(flask_app, host='0.0.0.0', port=8080)
    #flask_app.run(debug=False,port=8090)