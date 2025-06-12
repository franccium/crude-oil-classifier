
from training.model_definition import models
from utils.plots import plot_decision_boundary, plot_svc_decision_boundary, plot_decision_tree, plot_random_forest_tree

class ModelsTraining:
    def __init__(self, X_train, y_train, X_test, y_test,
                 X, label_mapping):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.x = X
        self.label_mapping = label_mapping

    def mlp(self):
        mlp_model = models[0][1]
        mlp_model.fit(self.X_train, self.y_train)
        plot_decision_boundary(
            mlp_model, self.X_train, self.y_train.values, self.X_test,
            self.y_test.values,self.label_mapping,
            self.x.columns[0], self.x.columns[1], "MLP Decision Boundary"
        )

    def knn(self):
        knn_model = models[1][1]
        knn_model.fit(self.X_train, self.y_train)
        plot_decision_boundary(
            knn_model, self.X_train, self.y_train.values,
            self.X_test, self.y_test.values, self.label_mapping,
            self.x.columns[0], self.x.columns[1], "KNN Decision Boundary"
        )

    def svc(self):
        svc_model = models[2][1]
        plot_svc_decision_boundary(
            svc_model, self.X_train, self.y_train.values,
            self.X_test, self.y_test.values,
            self.label_mapping, self.x.columns[0], self.x.columns[1],
        )

    def decision_tree(self):
        dt_model = models[3][1]
        dt_model.fit(self.X_train, self.y_train)
        plot_decision_tree(
            dt_model, self.x.columns,
            list(self.label_mapping.keys()),
            "Decision Tree Visualization"
        )

    def random_forest(self):
        rf_model = models[4][1]
        rf_model.fit(self.X_train, self.y_train)
        plot_random_forest_tree(
            rf_model, self.x.columns,
            list(self.label_mapping.keys()),
            "Visualization of One Tree from the Random Forest"
        )
