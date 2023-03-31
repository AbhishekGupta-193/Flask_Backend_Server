import pickle #saves the model

import json
from flask import Flask, request #flask is the server for hosting
import numpy as np
from flask_cors import CORS #for permission access


# Declare a flask app
app = Flask(__name__)
CORS(app)

filename = 'model/diabetes_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# @app.route('/',methods=['GET'])
# def hello():
#     return "hello"

@app.route('/predict',methods=['POST'])
def predict():
    inp=request.get_json()
    # print(inp)
    input_data = (inp['preg'],inp['gl'],inp['bp'],inp['skv'],inp['il'],inp['bmi'],inp['dofv'],inp['age'])

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    

if __name__ == '__main__':
    app.run(host='0.0.0.0')
