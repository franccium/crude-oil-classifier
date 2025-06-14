import csv
from utils.data import load_data, label_mapping, reverse_label_mapping
import ui.state as state
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from training.cross_validation import evaluate_models_cv, report_cv_results
from training.model_definition import models
from training.models import ModelsTraining
import matplotlib.pyplot as plt

def load_sample_ids(selected_file):
    sample_ids = []
    with open(f"./data/{selected_file}", newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            if row:
                sample_ids.append(row[0])
    return sorted(set(sample_ids))

def clear_window(window):
    for widget in window.winfo_children():
        widget.destroy()

def ranking():
    graph_flags = state.graph_flags
    X, y, df = load_data(state.selected_file, graph_flags, state.featureset)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    all_rankings = {}
    all_class_acc = {}

    print('here2')

    if graph_flags['cv_summary']:
        all_rankings, all_class_acc = evaluate_models_cv(
            models=models,
            X=X,
            y=y,
            split_counts=[5, 6, 7],
            label_mapping=label_mapping,
            reverse_label_mapping=reverse_label_mapping
        )

    if graph_flags['feature_space']:
        if X.shape[1] >= 2:
            X_vis = X.iloc[:, :2]
            X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
                X_vis, y, test_size=0.2, random_state=9, stratify=y
            )
            scaler_vis = StandardScaler()
            X_train_vis_scaled = scaler_vis.fit_transform(X_train_vis)
            X_test_vis_scaled = scaler_vis.transform(X_test_vis)

            models_training = ModelsTraining(
                X_train_vis_scaled,
                y_train_vis,
                X_test_vis_scaled,
                y_test_vis,
                X_vis,
                label_mapping
            )
            models_training.mlp()
            models_training.knn()
            models_training.svc()
            models_training.decision_tree()
            models_training.random_forest()
        else:
            print("Not enough features for 2D decision boundary plots.")

        plt.show()

    if graph_flags['cv_summary']:
        report_cv_results(
            models,
            split_counts=[5, 6, 7],
            all_rankings=all_rankings,
            all_class_acc=all_class_acc,
            label_mapping=label_mapping
        )

    plt.show()