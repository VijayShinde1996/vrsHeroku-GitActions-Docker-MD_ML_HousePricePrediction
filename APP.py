from flask import Flask
import numpy as np
import pandas as pd
import joblib

APP = Flask(__name__)
scaling = joblib.load('scaling_md15.pkl')
pickle_file = joblib.load('train_md15.pkl')

if __name__ == '__main__':
    APP.run(debug=True)