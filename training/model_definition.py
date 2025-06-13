from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

models = [
    ("MLP", make_pipeline(StandardScaler(), MLPClassifier(
        activation='tanh', alpha=0.0001, hidden_layer_sizes=(100, 50),
        learning_rate='constant', max_iter=200, solver='lbfgs', random_state=9))), # max_iter = 2000 is needed for augmented data all features to coverge
    ("KNN", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=4, metric="manhattan", weights="uniform"))),
    ("SVC", make_pipeline(StandardScaler(), SVC(kernel='linear', C=100, random_state=9))),
    ("Decision Tree", DecisionTreeClassifier(
        criterion='gini', max_depth=3, max_features='sqrt',
        min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=9)),
    ("Random Forest", RandomForestClassifier(
        n_estimators=100, criterion='gini', max_depth=3, max_features='sqrt',
        min_samples_leaf=1, min_samples_split=2, random_state=9))
]
