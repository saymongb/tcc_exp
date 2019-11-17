from waitress import serve
import appPredictions
serve(appPredictions.flask_app, host='0.0.0.0', port=8090)