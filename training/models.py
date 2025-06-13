
from training.model_definition import models
from utils.plots import plot_decision_boundary, plot_decision_tree, plot_random_forest_tree
import joblib
from datetime import datetime
import os

class ModelsTraining:
    def __init__(self, X_train, y_train, X_test, y_test,
                 X, label_mapping):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.x = X
        self.label_mapping = label_mapping
        
    def _export_model(self, model, name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.pkl"
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        filepath = os.path.join(export_dir, filename)
        joblib.dump(model, filepath)
        print(f"Model exported to {filepath}")    

    def mlp(self, export=False):
        mlp_model = models[0][1]
        mlp_model.fit(self.X_train, self.y_train)
        if export:
            self._export_model(mlp_model, "mlp")
        plot_decision_boundary(
            mlp_model, self.X_train, self.y_train.values, self.X_test,
            self.y_test.values,self.label_mapping,
            self.x.columns[0], self.x.columns[1], "MLP Decision Boundary"
        )

    def knn(self, export=False):
        knn_model = models[1][1]
        knn_model.fit(self.X_train, self.y_train)
        if export:
            self._export_model(knn_model, "knn")
        plot_decision_boundary(
            knn_model, self.X_train, self.y_train.values,
            self.X_test, self.y_test.values, self.label_mapping,
            self.x.columns[0], self.x.columns[1], "KNN Decision Boundary"
        )

    def svc(self, export=False):
        svc_model = models[2][1]
        svc_model.fit(self.X_train, self.y_train)
        if export:
            self._export_model(svc_model, "svc")
        plot_decision_boundary(
            svc_model, self.X_train, self.y_train.values,
            self.X_test, self.y_test.values,
            self.label_mapping, self.x.columns[0], self.x.columns[1], "SVC Decision Boundary"
        )

    def decision_tree(self, export=False):
        dt_model = models[3][1]
        dt_model.fit(self.X_train, self.y_train)
        if export:
            self._export_model(dt_model, "decision_tree")
        plot_decision_tree(
            dt_model, self.x.columns,
            list(self.label_mapping.keys()),
            "Decision Tree Visualization"
        )

    def random_forest(self, export=False):
        rf_model = models[4][1]
        rf_model.fit(self.X_train, self.y_train)
        if export:
            self._export_model(rf_model, "random_forest")
        plot_random_forest_tree(
            rf_model, self.x.columns,
            list(self.label_mapping.keys()),
            "Visualization of One Tree from the Random Forest"
        )
