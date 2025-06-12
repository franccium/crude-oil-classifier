
import pandas as pd
import os
import joblib
from utils.markers import asses_ci

def test_sara_regression(S, Ar, R, As):
    input_data = pd.DataFrame([{
        'S': S,
        'Ar': Ar,
        'R': R,
        'As': As
    }])
    sara_reg_model = joblib.load(os.path.join('..', 'models', 'sara_linear_regression.pkl'))
    cii = sara_reg_model.predict(input_data)

    return asses_ci(cii), cii

