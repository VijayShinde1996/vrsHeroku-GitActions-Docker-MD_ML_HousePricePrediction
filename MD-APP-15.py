from flask import Flask,request,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import joblib

APP15 = Flask(__name__)
scalar = joblib.load('scaling_md15.pkl')
model = joblib.load('train_md15.pkl')

@APP15.route('/')
def home():
    return render_template('home.html')

@APP15.route('/predict',methods=['POST'])
def predict():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == '__main__':
    APP15.run(debug=True)