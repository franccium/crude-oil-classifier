
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from training.model_definition import models
from training.models import ModelsTraining
from training.cross_validation import evaluate_models_cv, report_cv_results
from utils.data import load_data, label_mapping, reverse_label_mapping
from ui.initial_selection import select_dataset, select_graphs, select_featureset, select_mode, select_model_file, get_best_model_for_featureset

from regression.train_regression_models import tsi_value_linear_regression_train, s_value_linear_regression_train, p_value_linear_regression_train, asmix_linear_regression_train

#tsi_value_linear_regression_train()
#p_value_linear_regression_train()
#s_value_linear_regression_train()
#asmix_linear_regression_train()

def test_exported_model():
    model_file = select_model_file()
    if not model_file:
        print("No model selected.")
        return
    featureset = select_featureset()
    if featureset is None:
        return
    filename = select_dataset()
    if filename is None:
        return
    flags = {
        'feature_space': False,
        'data_scatter': False,
        'cv_summary': False
    }
    graph_flags = {k: v for k, v in flags.items()}

    X, y, df = load_data(filename, graph_flags, featureset)
    import joblib
    model = joblib.load(model_file)
    y_pred = model.predict(X)
    from sklearn.metrics import accuracy_score, classification_report
    print("Test accuracy:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))

mode = select_mode()
if mode == "test":
    test_exported_model()
    exit()


filename = select_dataset()
if filename is None:
    exit()
featureset = select_featureset()
if featureset is None:
    exit()
    
best_model_for_featureset = get_best_model_for_featureset(featureset)

graph_flags = select_graphs()
X, y, df = load_data(filename, graph_flags, featureset)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
all_rankings = {}
all_class_acc = {}

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
