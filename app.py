import pickle #saves the model

import json
from flask import Flask, request #flask is the server for hosting
import numpy as np
from flask_cors import CORS #for permission access


# Declare a flask app
app = Flask(__name__)
CORS(app)

filename1 = 'model/diabetes_model.sav'
loaded_model1 = pickle.load(open(filename1, 'rb'))

filename2 = 'model/heart_disease_model.sav'
loaded_model2 = pickle.load(open(filename2, 'rb'))

filename3 = 'model/parkinsons_model.sav'
loaded_model3 = pickle.load(open(filename3, 'rb'))


@app.route('/predict1',methods=['POST'])
def predict1():
    inp=request.get_json()
    # print(inp)
    input_data1 = (inp['preg'],inp['gl'],inp['bp'],inp['skv'],inp['il'],inp['bmi'],inp['dofv'],inp['age'])

    # changing the input_data to numpy array
    input_data_as_numpy_array1 = np.asarray(input_data1)

    # reshape the array as we are predicting for one instance
    input_data_reshaped1 = input_data_as_numpy_array1.reshape(1,-1)

    prediction1 = loaded_model1.predict(input_data_reshaped1)
    print(prediction1)

    if (prediction1[0] == 0):
        return '0: The person is not diabetic'
    else:
        return '1: The person is diabetic'
    
@app.route('/predict2',methods=['POST'])
def predict2():
    inp=request.get_json()
    # print(inp)
    input_data1 = ( int(inp['age']),int( inp['sex']), int( inp['cpt']), int(inp['rpb']), int( inp['sc']), int( inp['fbg']), int( inp['rer']), int(inp['mhra']), int( inp['eia']), int( inp['stdpi']), int( inp['spest']), int( inp['mvcf']), int(inp['tnfr']))

    # changing the input_data to numpy array
    input_data_as_numpy_array1 = np.asarray(input_data1)

    # reshape the array as we are predicting for one instance
    input_data_reshaped1 = input_data_as_numpy_array1.reshape(1,-1)

    prediction1 = loaded_model2.predict(input_data_reshaped1)
    print(prediction1)

    if (prediction1[0] == 0):
        return '0: The person is not a heart patient '
    else:
        return '1: The person is a heart patient '
    
@app.route('/predict3',methods=['POST'])
def predict3():
    inp=request.get_json()
    # print(inp)
    input_data1 = (inp['m1'],inp['m2'],inp['m3'],inp['m4'],inp['m5'],inp['m6'],inp['m7'],inp['jd'],inp['m8'],inp['m9'],inp['sa3'],inp['sa5'],inp['m10'],inp['sd'],inp['nhr'],inp['hnr'],inp['rpde'],inp['dfa'],inp['s1'],inp['s2'],inp['d2'],inp['ppe'])

    # changing the input_data to numpy array
    input_data_as_numpy_array1 = np.asarray(input_data1)

    # reshape the array as we are predicting for one instance
    input_data_reshaped1 = input_data_as_numpy_array1.reshape(1,-1)

    prediction1 = loaded_model3.predict(input_data_reshaped1)
    print(prediction1)

    if (prediction1[0] == 0):
        return '0: The person is having parkinson disease '
    else:
        return '1: The person is not having parkinson disease '
    

if __name__ == '__main__':
    app.run(host='0.0.0.0')
