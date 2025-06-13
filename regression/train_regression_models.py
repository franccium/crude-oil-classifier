
import joblib
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from utils.parsers import parse_sara, parse_s_value
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
    joblib.dump(regr, os.path.join('..', 'models', 'sara_linear_regression.pkl' ))

    return regr

def s_value_linear_regression_train():
    df = parse_s_value('s_value.csv')
    X = df.drop(columns=['S_Value_res'])
    y = df['S_Value_res']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regr = LinearRegression()
    regr.fit(X_train, y_train)
#    joblib.dump(regr, os.path.join('..', 'models', 's_value_linear_regression.pkl' ))
    train_score = regr.score(X_train, y_train)
    test_score = regr.score(X_test, y_test)
    print(f"Train R^2 Score: {train_score:.4f}")
    print(f"Test R^2 Score: {test_score:.4f}")

    # Predict on test set
    y_pred = regr.predict(X_test)

    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal fit')
    plt.xlabel('Actual S_Value_res')
    plt.ylabel('Predicted S_Value_res')
    plt.title('Linear Regression: Actual vs Predicted on Test Set')
    plt.legend()
    plt.grid(True)
    plt.show()

