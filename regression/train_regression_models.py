
import joblib
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from utils.parsers import parse_sara, parse_s_value, parse_tsi_value, parse_p_value
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

def linear_regression_train(parse_func, target_column: str, title_prefix: str = ""):
    df = parse_func()
    print(df.head())
    X = df.drop(columns=[target_column])
    y = df[target_column]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        y,
        c='green', marker='o'
    )
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])
    ax.set_zlabel(target_column)
    ax.set_title(f'3D Scatter Plot: {X.columns[0]} vs {X.columns[1]} vs {target_column}')
    plt.tight_layout()
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regr = LinearRegression()
    regr.fit(X_train, y_train)
    train_score = regr.score(X_train, y_train)
    test_score = regr.score(X_test, y_test)
    print(f"Train R^2 Score: {train_score:.4f}")
    print(f"Test R^2 Score: {test_score:.4f}")

    y_pred = regr.predict(X_test)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal fit')
    plt.xlabel(f'Actual {target_column}')
    plt.ylabel(f'Predicted {target_column}')
    plt.title(f'{title_prefix} Linear Regression: Actual vs Predicted on Test Set')
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_regressors_with_gridsearch(parse_func, target_column: str, title_prefix: str = ""):
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import Ridge, BayesianRidge, LinearRegression, HuberRegressor
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.svm import SVR, LinearSVR, NuSVR
    df = parse_func()
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models_and_params = [
        ("AdaBoostRegressor", AdaBoostRegressor(random_state=42), {
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
            "loss": ["linear", "square", "exponential"]
        }),
        ("GradientBoostingRegressor", GradientBoostingRegressor(random_state=42), {
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [2, 3, 4, 6, 8],
            "subsample": [0.7, 0.85, 1.0],
            "min_samples_split": [2, 5, 10]
        }),
        ("DecisionTreeRegressor", DecisionTreeRegressor(random_state=42), {
            "max_depth": [2, 4, 6, 8, 10, 12, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "criterion": ["squared_error", "friedman_mse", "absolute_error"]
        }),
        ("MLPRegressor", MLPRegressor(max_iter=2000, random_state=42), {
            "hidden_layer_sizes": [(50,), (100,), (50,50), (100,50), (100,100)],
            "activation": ["relu", "tanh", "logistic"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "adaptive"]
        }),
        ("ExtraTreesRegressor", ExtraTreesRegressor(random_state=42), {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [None, 4, 8, 12],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }),
        ("Ridge", Ridge(), {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            "solver": ["auto", "svd", "cholesky", "lsqr", "sag"]
        }),
        ("BayesianRidge", BayesianRidge(), {
            "alpha_1": [1e-6, 1e-5, 1e-4, 1e-3],
            "alpha_2": [1e-6, 1e-5, 1e-4, 1e-3],
            "lambda_1": [1e-6, 1e-5, 1e-4, 1e-3],
            "lambda_2": [1e-6, 1e-5, 1e-4, 1e-3]
        }),
        ("LinearRegression", LinearRegression(), {}),  # No hyperparameters
        ("PLSRegression", PLSRegression(), {
            "n_components": [1, 2, 3, 4, 5]
        }),
        ("HuberRegressor", HuberRegressor(), {
            "epsilon": [1.1, 1.35, 1.5, 1.75, 2.0],
            "alpha": [0.0001, 0.001, 0.01]
        }),
        ("NuSVR", NuSVR(), {
            "C": [0.1, 1, 10],
            "nu": [0.25, 0.5, 0.75],
            "kernel": ["rbf", "linear"]
        }),
    ]

    results = []
    for name, model, params in models_and_params:
        try:
            if params:
                grid = GridSearchCV(model, params, cv=3, scoring='r2', n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                model.fit(X_train, y_train)
                best_model = model
                best_params = {}
            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            results.append((name, r2, rmse, best_params))
            print(f"{name}: Test R^2 = {r2:.4f}, RMSE = {rmse:.4f}, Best Params: {best_params}")
        except Exception as e:
            print(f"{name}: Failed with error: {e}")

    results.sort(key=lambda x: x[1], reverse=True)
    print("\n====== Model Ranking by Test R^2 ======")
    for i, (name, r2, rmse, params) in enumerate(results, 1):
        print(f"{i}. {name}: R^2 = {r2:.4f}, RMSE = {rmse:.4f}, Best Params: {params}")

    # Plot best model
    if results:
        best_name, _, _, best_params = results[0]
        best_model = dict((n, m) for n, m, _ in models_and_params)[best_name].set_params(**best_params)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit')
        plt.xlabel(f'Actual {target_column}')
        plt.ylabel(f'Predicted {target_column}')
        plt.title(f'{title_prefix} {best_name}: Actual vs Predicted')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
def compare_regressors(parse_func, target_column: str, title_prefix: str = "", n_runs: int = 50):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, SGDRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR, LinearSVR, NuSVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor

    df = parse_func()
    X = df.drop(columns=[target_column])
    y = df[target_column]

    regressors = [
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge()),
        ("Lasso", Lasso()),
        ("ElasticNet", ElasticNet()),
        ("BayesianRidge", BayesianRidge()),
        ("HuberRegressor", HuberRegressor()),
        ("SGDRegressor", SGDRegressor(max_iter=1000, tol=1e-3)),
        ("DecisionTreeRegressor", DecisionTreeRegressor(random_state=42)),
        ("RandomForestRegressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("ExtraTreesRegressor", ExtraTreesRegressor(n_estimators=100, random_state=42)),
        ("GradientBoostingRegressor", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("AdaBoostRegressor", AdaBoostRegressor(n_estimators=100, random_state=42)),
        ("BaggingRegressor", BaggingRegressor(n_estimators=100, random_state=42)),
        ("KNeighborsRegressor", KNeighborsRegressor()),
        ("NuSVR", NuSVR()),
        ("RadiusNeighborsRegressor", RadiusNeighborsRegressor()),
        ("GaussianProcessRegressor", GaussianProcessRegressor()),
        ("PLSRegression", PLSRegression()),
        ("LinearSVR", LinearSVR()),
        ("SVR", SVR()),
        ("MLPRegressor", MLPRegressor(max_iter=1000, random_state=42)),
    ]

    results = {name: {"r2": [], "rmse": []} for name, _ in regressors}

    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=run)
        for name, regr in regressors:
            try:
                regr.fit(X_train, y_train)
                y_pred = regr.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                results[name]["r2"].append(r2)
                results[name]["rmse"].append(rmse)
            except Exception:
                continue

    summary = []
    for name in results:
        r2_mean = np.mean(results[name]["r2"])
        rmse_mean = np.mean(results[name]["rmse"])
        summary.append((name, r2_mean, rmse_mean))
        print(f"{name}: Mean Test R^2 = {r2_mean:.4f}, Mean RMSE = {rmse_mean:.4f}")

    summary.sort(key=lambda x: x[1], reverse=True)
    print("\n====== Model Ranking by Mean Test R^2 ======")
    for i, (name, r2, rmse) in enumerate(summary, 1):
        print(f"{i}. {name}: Mean R^2 = {r2:.4f}, Mean RMSE = {rmse:.4f}")

    best_name = summary[0][0]
    best_regr = dict(regressors)[best_name]
    best_regr.fit(X_train, y_train)
    y_pred = best_regr.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit')
    plt.xlabel(f'Actual {target_column}')
    plt.ylabel(f'Predicted {target_column}')
    plt.title(f'{title_prefix} {best_name}: Actual vs Predicted (Last Run)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_best_regressors(parse_func, target_column: str, title_prefix: str = "", n_runs: int = 40):
    df = parse_func()
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
    from sklearn.linear_model import HuberRegressor, BayesianRidge, LinearRegression, Ridge
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR, LinearSVR, NuSVR

    regressors = [
        ("AdaBoostRegressor", AdaBoostRegressor(
            learning_rate=1.0, loss='linear', n_estimators=300, random_state=42)),
        ("MLPRegressor", MLPRegressor(
            activation='relu', alpha=0.01, hidden_layer_sizes=(100, 100), learning_rate='constant', max_iter=1000, random_state=42)),
        ("BayesianRidge", BayesianRidge(
            alpha_1=1e-06, alpha_2=0.001, lambda_1=0.001, lambda_2=1e-06)),
        ("Ridge", Ridge(
            alpha=0.01, solver='lsqr')),
        ("HuberRegressor", HuberRegressor(
            alpha=0.01, epsilon=1.1)),
        ("GradientBoostingRegressor", GradientBoostingRegressor(
            learning_rate=0.2, max_depth=3, min_samples_split=10, n_estimators=50, subsample=1.0, random_state=42)),
        ("LinearRegression", LinearRegression()),
        ("PLSRegression", PLSRegression(
            n_components=4)),
        ("NuSVR", NuSVR(
            C=10, kernel='rbf', nu=0.25)),
    ]

    results = {name: {"r2": [], "rmse": []} for name, _ in regressors}

    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=run)
        for name, regr in regressors:
            try:
                regr.fit(X_train, y_train)
                y_pred = regr.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                results[name]["r2"].append(r2)
                results[name]["rmse"].append(rmse)
            except Exception:
                continue

    summary = []
    for name in results:
        r2_mean = np.mean(results[name]["r2"])
        rmse_mean = np.mean(results[name]["rmse"])
        summary.append((name, r2_mean, rmse_mean))
        print(f"{name}: Mean Test R^2 = {r2_mean:.4f}, Mean RMSE = {rmse_mean:.4f}")

    summary.sort(key=lambda x: x[1], reverse=True)
    print("\n====== Model Ranking by Mean Test R^2 ======")
    for i, (name, r2, rmse) in enumerate(summary, 1):
        print(f"{i}. {name}: Mean R^2 = {r2:.4f}, Mean RMSE = {rmse:.4f}")

    best_name = summary[0][0]
    best_regr = dict(regressors)[best_name]
    best_regr.fit(X_train, y_train)
    y_pred = best_regr.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit')
    plt.xlabel(f'Actual {target_column}')
    plt.ylabel(f'Predicted {target_column}')
    plt.title(f'{title_prefix} {best_name}: Actual vs Predicted (Last Run)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def p_value_linear_regression_train():
    #linear_regression_train(parse_p_value, 'P_Value_res', 'P')
    compare_regressors(parse_p_value, 'P_Value_res', 'P')
    #compare_regressors_with_gridsearch(parse_p_value, 'P_Value_res', 'P')

def tsi_value_linear_regression_train():
    compare_regressors(parse_tsi_value, 'TSI_Value_res', 'TSI')
    #linear_regression_train(parse_tsi_value, 'TSI_Value_res', 'TSI')
    
def asmix_linear_regression_train():
    from utils.parsers import parse_asmix, parse_asmix_with_density, parse_asmix_with_density_find_CII
    target = 'AsMix'
    target = 'CII'
    
    #compare_regressors(parse_asmix_with_density, target, target)
    #compare_regressors(parse_asmix_with_density_find_CII, target, target)
    #compare_regressors_with_gridsearch(parse_asmix_with_density_find_CII, target, target)
    compare_best_regressors(parse_asmix_with_density_find_CII, target, target)
    #compare_regressors_with_gridsearch(parse_asmix_with_density, target, target)
    #compare_best_regressors(parse_asmix_with_density, target, target)

def s_value_linear_regression_train():
    #compare_regressors_with_gridsearch(parse_s_value, 'S_Value_res', 'S')
    #compare_regressors(parse_s_value, 'S_Value_res', 'S')
    df = parse_tsi_value()
    print(df.head())
    df = parse_s_value('s_value.csv')
    print(df.head())
    X = df.drop(columns=['S_Value_res'])
    y = df['S_Value_res']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        X.iloc[:, 0],  # First feature (X-axis)
        X.iloc[:, 1],  # Second feature (Y-axis)
        y,  # Target (Z-axis)
        c='green', marker='o'
    )
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])
    ax.set_zlabel('S_Value_res')
    ax.set_title('3D Scatter Plot: Feature1 vs Feature2 vs Target')
    plt.tight_layout()
    plt.show()

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

