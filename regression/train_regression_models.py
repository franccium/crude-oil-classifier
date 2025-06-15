
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.svm import NuSVR
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils.parsers import parse_s_value, parse_tsi_value, parse_p_value, parse_asmix_with_density_find_CII
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

def tune_top_regressors(parse_func, target_column: str, cv_folds: int = 5, n_repeats: int = 5):
    df = parse_func()
    X = df.drop(columns=[target_column])
    y = df[target_column]

    rkf = RepeatedKFold(n_splits=cv_folds, n_repeats=n_repeats, random_state=9)

    param_grids = {
        "ExtraTreesRegressor": {
            #"n_estimators": [50, 100, 200, 300],
            "n_estimators": [50, 100],
            "max_depth": [None, 4, 8],
            "min_samples_split": [2, 5, 8],
            "min_samples_leaf": [1, 2, 4]
        }
    }

    regressors = {
        "ExtraTreesRegressor": ExtraTreesRegressor(random_state=9)
    }

    for name in regressors:
        print(f"\nTuning {name}...")
        grid = GridSearchCV(
            regressors[name],
            param_grids[name],
            scoring='r2',
            cv=rkf,
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X, y)
        print(f"Best R^2: {grid.best_score_:.4f}")
        print(f"Best Params: {grid.best_params_}")

        from sklearn.model_selection import cross_val_score
        best_model = grid.best_estimator_
        neg_mse = cross_val_score(best_model, X, y, cv=rkf, scoring='neg_mean_squared_error', n_jobs=-1)
        rmse = np.sqrt(-neg_mse)
        print(f"Best RMSE (mean ± std): {rmse.mean():.4f} ± {rmse.std():.4f}")

def compare_regressors_repeated_cv(parse_func, target_column: str, title_prefix: str = "", cv_folds: int = 5, n_repeats: int = 25):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR, LinearSVR, NuSVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.cross_decomposition import PLSRegression
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
        ("DecisionTreeRegressor", DecisionTreeRegressor(random_state=9)),
        ("RandomForestRegressor", RandomForestRegressor(n_estimators=100, random_state=9)),
        ("ExtraTreesRegressor", ExtraTreesRegressor(n_estimators=100, random_state=9)),
        ("GradientBoostingRegressor", GradientBoostingRegressor(n_estimators=100, random_state=9)),
        ("AdaBoostRegressor", AdaBoostRegressor(n_estimators=100, random_state=9)),
        ("BaggingRegressor", BaggingRegressor(n_estimators=100, random_state=9)),
        ("KNeighborsRegressor", KNeighborsRegressor()),
        ("NuSVR", NuSVR()),
        ("RadiusNeighborsRegressor", RadiusNeighborsRegressor()),
        ("PLSRegression", PLSRegression()),
        ("LinearSVR", LinearSVR()),
        ("SVR", SVR()),
        ("MLPRegressor", MLPRegressor(max_iter=1000, random_state=9)),
    ]

    rkf = RepeatedKFold(n_splits=cv_folds, n_repeats=n_repeats, random_state=9)
    results = []

    for name, regr in regressors:
        try:
            r2_scores = cross_val_score(regr, X, y, cv=rkf, scoring='r2', n_jobs=-1)
            neg_mse_scores = cross_val_score(regr, X, y, cv=rkf, scoring='neg_mean_squared_error', n_jobs=-1)
            rmse_scores = np.sqrt(-neg_mse_scores)
            results.append((
                name,
                np.mean(r2_scores), np.std(r2_scores),
                np.mean(rmse_scores), np.std(rmse_scores)
            ))
            print(
                f"{name}: Repeated CV Mean R^2 = {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f}), "
                f"Mean RMSE = {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})"
            )
        except Exception as e:
            print(f"{name}: Failed with error: {e}")

    results.sort(key=lambda x: x[1], reverse=True)
    print("\n====== Model Ranking by Repeated CV Mean R^2 ======")
    for i, (name, r2_mean, r2_std, rmse_mean, rmse_std) in enumerate(results, 1):
        print(
            f"{i}. {name}: Mean R^2 = {r2_mean:.4f} (±{r2_std:.4f}), "
            f"Mean RMSE = {rmse_mean:.4f} (±{rmse_std:.4f})"
        )

    names = [x[0] for x in results]
    mean_r2 = [x[1] for x in results]
    std_r2 = [x[2] for x in results]
    plt.figure(figsize=(12, 5))
    plt.bar(names, mean_r2, yerr=std_r2, color='skyblue', capsize=5)
    plt.ylabel('Repeated CV Mean R²')
    plt.title('Repeated CV Mean R² by Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
def analyze_repeated_cv_model(model, parse_func, target_column: str, cv_folds: int = 5, n_repeats: int = 10):
    df = parse_func()
    X = df.drop(columns=[target_column])
    y = df[target_column]

    rkf = RepeatedKFold(n_splits=cv_folds, n_repeats=n_repeats, random_state=9)

    r2_scores = cross_val_score(model, X, y, cv=rkf, scoring='r2', n_jobs=-1)
    neg_mse_scores = cross_val_score(model, X, y, cv=rkf, scoring='neg_mean_squared_error', n_jobs=-1)
    rmse_scores = np.sqrt(-neg_mse_scores)

    print(f"Repeated CV results:")
    print(f"Mean R^2: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"Mean RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")

    plt.figure(figsize=(8, 4))
    plt.hist(r2_scores, bins=15, color='skyblue', edgecolor='k')
    plt.title('Distribution of R² Scores (Repeated CV)')
    plt.xlabel('R² Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(rmse_scores, bins=15, color='salmon', edgecolor='k')
    plt.title('Distribution of RMSE Scores (Repeated CV)')
    plt.xlabel('RMSE')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    model.fit(X, y)
    y_pred = model.predict(X)
    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred, alpha=0.6, color='blue', label='Predicted vs Actual')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('ExtraTreesRegressor: Actual vs Predicted (All Data)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    errors = y - y_pred
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=30, color='purple', alpha=0.7)
    plt.xlabel('Prediction Error (y_true - y_pred)')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Histogram (All Data)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def export_ensemble_nusvr(parse_func, target_column: str, n_models: int = 5, name: str = "model.pkl"):
    from sklearn.svm import NuSVR
    df = parse_func()
    X = df.drop(columns=[target_column])
    y = df[target_column]

    rkf = RepeatedKFold(n_splits=5, n_repeats=n_models, random_state=9)
    ensemble = []
    for i, (train_idx, _) in enumerate(rkf.split(X)):
        if i >= n_models:
            break
        model = NuSVR(C=1, kernel='rbf', nu=0.5)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        ensemble.append(model)
    path = os.path.join("models", name)
    joblib.dump(ensemble, path)
    print(f"Ensemble of {n_models} NuSVR models exported to {path}")

def analyze_tree_ensemble(model, parse_func, target_column: str = 'TSI_Value_res', title_prefix: str = "TSI", n_models: int = 10, radius: float = 1.0):
    df = parse_func()
    X = df.drop(columns=[target_column])
    y = df[target_column].reset_index(drop=True)

    preds = np.zeros((n_models, len(y)))
    r2_list = []
    rmse_list = []

    for i in range(n_models):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        fill_value = y_train.mean()
        model.fit(X_train, y_train)
        preds[i] = model.predict(X)
        y_pred_test = model.predict(X_test)
        y_pred_test = np.where(np.isnan(y_pred_test), fill_value, y_pred_test)
        r2_list.append(r2_score(y_test, y_pred_test))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))

    ensemble_pred = preds.mean(axis=0)
    ensemble_pred = np.where(np.isnan(ensemble_pred), y.mean(), ensemble_pred)
    r2_ensemble = r2_score(y, ensemble_pred)
    rmse_ensemble = np.sqrt(mean_squared_error(y, ensemble_pred))

    print(f"Ensemble R^2 (all data): {r2_ensemble:.4f}")
    print(f"Ensemble RMSE (all data): {rmse_ensemble:.4f}")
    print(f"Mean single-model R^2: {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
    print(f"Mean single-model RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y, ensemble_pred, alpha=0.6, color='blue', label='Ensemble Predicted vs Actual')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit')
    plt.xlabel('Actual TSI')
    plt.ylabel('Ensemble Predicted TSI')
    plt.title(f'{title_prefix} RadiusNeighborsRegressor Ensemble: Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    errors = y - ensemble_pred
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=30, color='purple', alpha=0.7)
    plt.xlabel('Prediction Error (y_true - y_pred)')
    plt.ylabel('Frequency')
    plt.title(f'{title_prefix} RadiusNeighborsRegressor Ensemble: Prediction Error Histogram')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def p_value_linear_regression_train():
    model = ExtraTreesRegressor(
        max_depth=8,
        min_samples_leaf=1,
        min_samples_split=5,
        n_estimators=200
    )
    #compare_regressors_repeated_cv(parse_p_value, 'P_Value_res', 'P', cv_folds=5, n_repeats=60)
    #export_ensemble_model(model, parse_p_value, 'P_Value_res', n_models=10, name='p_value_ensemble7.pkl')
    analyze_tree_ensemble(model, parse_p_value, target_column='P_Value_res', title_prefix='P', n_models=10)

def tsi_value_linear_regression_train():
    model = ExtraTreesRegressor(
        max_depth=8,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=100
    )
    #compare_regressors_repeated_cv(parse_tsi_value, 'TSI_Value_res', 'TSI', cv_folds=5, n_repeats=60)
    #export_ensemble_model(model, parse_tsi_value, 'TSI_Value_res', n_models=10, name='tsi_value_ensemble7.pkl')
    analyze_tree_ensemble(model, parse_tsi_value, target_column='TSI_Value_res', title_prefix='TSI', n_models=10)

def asmix_linear_regression_train():
    model = NuSVR(C=1, kernel='rbf', nu=0.5)
    
    #compare_regressors_repeated_cv(parse_asmix_with_density_find_CII, 'CII', 'CII', cv_folds=5, n_repeats=60)
    #export_ensemble_model(model, parse_asmix_with_density_find_CII, 'CII', n_models=5, name='asmix_nusvr_ensemble.pkl')
    compare_regressors_repeated_cv(parse_asmix_with_density_find_CII, 'CII', 'CII', cv_folds=5, n_repeats=60)

def s_value_linear_regression_train():
    model = ExtraTreesRegressor(
        max_depth=8,
        min_samples_leaf=1,
        min_samples_split=5,
        n_estimators=200
    )
    #compare_regressors_repeated_cv(parse_s_value, 'S_Value_res', 'S', cv_folds=5, n_repeats=60)
    #export_ensemble_model(model, parse_s_value, 'S_Value_res', n_models=1, name='s_value_ensemble_invar.pkl')
    analyze_tree_ensemble(model, parse_s_value, target_column='S_Value_res', title_prefix='S', n_models=1)