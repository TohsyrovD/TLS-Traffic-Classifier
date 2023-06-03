import os
import pandas as pd
import dill as pickle
from flask import Flask, jsonify, request

clf = input('Введите модель:')

app = Flask(__name__)

@app.route('/')
def home_endpoint():
	return 'Hello World!'

@app.route('/predict', methods=['POST'])
def apicall():
	# Датафрейм pandas из API-вызова
	try:
		test_json = request.get_json()
		test = pd.read_json(test_json, orient='records')

	except Exception as e:
		raise e
	
	#clf = 'LDA_model.pkl'
	
	if test.empty:
		return(bad_request())
	else:
		# Загружаем модель в формате .pkl
		print("Loading the model...")
		loaded_model = None
		with open(clf, 'rb') as f:
			loaded_model = pickle.load(f)

		print("The model has been loaded... doing predictions now...")
		predictions = loaded_model.predict(test)

		# Добавим прогнозы в новый датафрейм pandas
		prediction_series = list(pd.Series(predictions))

		final_predictions = pd.DataFrame(list(zip(prediction_series)))
		
		# Определим код ответа и вернем json
		responses = jsonify(predictions=final_predictions.to_json(orient="records"))
		responses.status_code = 200
 
		return (responses)


@app.errorhandler(400)
def bad_request(error=None):
	message = {
			'status': 400,
			'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
	}
	resp = jsonify(message)
	resp.status_code = 400

	return resp

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8000)