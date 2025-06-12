
import joblib
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils.parsers import parse_sara
from utils.augmentation import sara_aug

def sara_linear_regression_train():
    df = parse_sara('sara.csv')
    X = df.drop(columns=['CII', 'ID_1', 'ID_2', '%_1', '%_2', 'Nr'])
    y = df['CII']

    augmented_df = sara_aug(df)
    augmented_x =  augmented_df.drop(columns=['CII'])
    augmented_y = augmented_df['CII']
    X = pd.concat([X, augmented_x], ignore_index=True)
    y = pd.concat([y, augmented_y], ignore_index=True)

    regr = LinearRegression()
    regr.fit(X, y)
    joblib.dump(regr, os.path.join('..', 'models', 'sara_linear_regression.plt' ))

    return regr

sara_linear_regression_train()
