import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from sklearn.tree import plot_tree
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import tkinter as tk
from tkinter import messagebox
from matplotlib.lines import Line2D

def select_dataset():
    def set_choice(choice):
        nonlocal selected_file
        selected_file = choice
        root.destroy()

    selected_file = None
    root = tk.Tk()
    root.title("Choose dataset")
    tk.Label(root, text="Choose dataset:").pack(padx=20, pady=10)
    tk.Button(root, text="Original", width=20, command=lambda: set_choice("data_original.csv")).pack(pady=5)
    tk.Button(root, text="Augmented", width=20, command=lambda: set_choice("data_augmented.csv")).pack(pady=5)
    root.mainloop()
    return "data\\" + selected_file

filename = select_dataset()
if filename is None:
    exit()

def select_graphs():
    root = tk.Tk()
    flags = {
        'feature_space': tk.BooleanVar(master=root, value=True),
        'data_scatter': tk.BooleanVar(master=root, value=True),
        'cv_summary': tk.BooleanVar(master=root, value=True)
    }
    def submit():
        root.destroy()
    root.title("Select Graphs to Show")
    tk.Label(root, text="Select which graphs to display:").pack(padx=20, pady=10)
    tk.Checkbutton(root, text="Show feature space visualizations", variable=flags['feature_space']).pack(anchor='w', padx=20)
    tk.Checkbutton(root, text="Show data scatter plots", variable=flags['data_scatter']).pack(anchor='w', padx=20)
    tk.Checkbutton(root, text="Show cross-validation summary graphs", variable=flags['cv_summary']).pack(anchor='w', padx=20)
    tk.Button(root, text="OK", command=submit).pack(pady=10)
    root.mainloop()
    return {k: v.get() for k, v in flags.items()}

graph_flags = select_graphs()

df = pd.read_csv(filename)
print("\n=== FULL DATASET ===")
print(df.to_string(index=False))
print("====================\n")
if 'Nr' in df.columns:
    df = df.drop('Nr', axis='columns')
label_mapping = {'light': 0, 'medium': 1, 'heavy': 2}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
df['Typ'] = df['Typ'].map(label_mapping)
X = df.drop(columns=['ID próbki', 'Typ', 'S (%)', 'Ar (%)', 'R (%)', 'As (%)'])
y = df['Typ']

custom_palette = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}
label_names = {0: 'light', 1: 'medium', 2: 'heavy'}

if graph_flags['data_scatter']:
    plt.figure(figsize=(8, 6))
    for typ, color in custom_palette.items():
        plt.scatter(
            df.loc[df['Typ'] == typ, 'Gęstość'],
            df.loc[df['Typ'] == typ, 'CII'],
            c=color,
            label=label_names[typ],
            edgecolor='k',
            s=50
        )
    plt.xlabel('Gęstość')
    plt.ylabel('CII')
    plt.legend(title='Type')
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for typ, color in custom_palette.items():
        ax.scatter(
            df.loc[df['Typ'] == typ, 'Gęstość'],
            df.loc[df['Typ'] == typ, 'CII'],
            df.loc[df['Typ'] == typ, 'S (%)'],
            c=color,
            label=label_names[typ],
            edgecolor='k',
            s=50
        )
    ax.set_xlabel('Gęstość')
    ax.set_ylabel('CII')
    ax.set_zlabel('S (%)')
    ax.legend(title='Type')
    plt.tight_layout()
    plt.show(block=False)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

split_counts = [5, 6, 7]
all_rankings = {}
all_class_acc = {}

models = [
    ("MLP", MLPClassifier(
        activation='tanh', alpha=0.0001, hidden_layer_sizes=(100, 50),
        learning_rate='constant', max_iter=200, solver='lbfgs', random_state=9)),
    ("KNN", KNeighborsClassifier(n_neighbors=4, metric="manhattan", weights="uniform")),
    ("SVC", SVC(kernel='linear', C=100, random_state=9)),
    ("Decision Tree", DecisionTreeClassifier(
        criterion='gini', max_depth=3, max_features='sqrt',
        min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=9)),
    ("Random Forest", RandomForestClassifier(
        n_estimators=100, criterion='gini', max_depth=3, max_features='sqrt',
        min_samples_leaf=1, min_samples_split=2, random_state=9))
]

for split in split_counts:
    print("\n" + "="*60)
    print(f" CROSS-VALIDATION SPLIT COUNT: {split} ".center(60, "="))
    print("="*60)
    model_scores = []
    class_acc = {}

    cv = StratifiedKFold(n_splits=split, shuffle=True, random_state=9)

    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    for name, model in models:
        print("="*50)
        print(name)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        mean_acc = cv_scores.mean()
        model_scores.append((name, mean_acc))
        y_pred_cv = cross_val_predict(model, X_scaled, y, cv=cv)
        # Map integer labels to string labels for reporting
        y_str = y.map(reverse_label_mapping)
        y_pred_cv_str = pd.Series(y_pred_cv).map(reverse_label_mapping)
        report = classification_report(y_str, y_pred_cv_str, target_names=label_mapping.keys(), output_dict=True)
        print(classification_report(y_str, y_pred_cv_str, target_names=label_mapping.keys()))
        # Collect per-class accuracy
        class_acc[name] = {cls: report[cls]['recall'] for cls in label_mapping.keys()}
        print(f"Mean CV accuracy: {mean_acc:.3f}")

    # Ranking for this split count
    print("\n" + "="*50)
    print(f"FINAL MODEL RANKING (CV splits: {split})")
    print("="*50)
    ranking = sorted(model_scores, key=lambda x: x[1], reverse=True)
    all_rankings[split] = ranking
    all_class_acc[split] = class_acc
    for i, (name, score) in enumerate(ranking, 1):
        print(f"{i}. {name:<15} {score:.3f}")
    print("="*50)

    # Per-class ranking for this split
    print("\nPer-class accuracy ranking:")
    for cls in label_mapping.keys():
        print(f"\nClass: {cls}")
        class_ranking = sorted([(name, class_acc[name][cls]) for name in class_acc], key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(class_ranking, 1):
            print(f"{i}. {name:<15} {score:.3f}")

def plot_decision_boundary(clf, X_train, y_train, X_test, y_test, title, label_mapping):
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    label_order = list(label_mapping.keys())
    custom_palette = {'lekkie': '#1f77b4', 'średnie': '#ff7f0e', 'ciężkie': '#2ca02c',
                     'light': '#1f77b4', 'medium': '#ff7f0e', 'heavy': '#2ca02c'}
    background_cmap = ListedColormap([custom_palette[label] for label in label_order])

    y_train_labels = [reverse_label_mapping[i] for i in y_train]
    y_test_labels = [reverse_label_mapping[i] for i in y_test]

    h = .02
    x_min = min(X_train[:, 0].min(), X_test[:, 0].min()) - .5
    x_max = max(X_train[:, 0].max(), X_test[:, 0].max()) + .5
    y_min = min(X_train[:, 1].min(), X_test[:, 1].min()) - .5
    y_max = max(X_train[:, 1].max(), X_test[:, 1].max()) + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=background_cmap, levels=np.arange(-0.5, len(label_order)+0.5, 1))
    sns.scatterplot(
        x=X_train[:, 0], y=X_train[:, 1], hue=y_train_labels,
        palette=custom_palette, hue_order=label_order,
        edgecolor="k", marker="o", s=50, legend=True
    )
    sns.scatterplot(
        x=X_test[:, 0], y=X_test[:, 1], hue=y_test_labels,
        palette=custom_palette, hue_order=label_order,
        edgecolor="w", marker="X", s=70, legend=False
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker='X', color='w', label='Test data (X marker)',
                          markerfacecolor='gray', markeredgecolor='w', markersize=10, linestyle='None'))
    plt.legend(handles=handles, title="Typ")
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.tight_layout()
    plt.show(block=False)

def plot_decision_tree(clf, feature_names, class_names, title):
    plt.figure(figsize=(20, 10))
    plot_tree(
        clf,
        filled=True,
        feature_names=feature_names,
        class_names=class_names,
        rounded=True,
        fontsize=10
    )
    plt.title(title)
    plt.tight_layout()
    plt.show(block=False)

def plot_random_forest_tree(rf, feature_names, class_names, title):
    plt.figure(figsize=(20, 10))
    plot_tree(
        rf.estimators_[0],
        filled=True,
        rounded=True,
        feature_names=feature_names,
        class_names=class_names,
        fontsize=10
    )
    plt.title(title)
    plt.tight_layout()
    plt.show(block=False)

def plot_svc_decision_boundary(clf, X_train, y_train, X_test, y_test, label_mapping):
    scaler_visualize = StandardScaler()
    X_train_s = scaler_visualize.fit_transform(X_train)
    X_test_s = scaler_visualize.transform(X_test)
    clf.fit(X_train_s, y_train)
    x_min, x_max = X_train_s[:, 0].min() - 1, X_train_s[:, 0].max() + 1
    y_min = X_train_s[:, 1].min() - 1
    y_max = X_train_s[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    y_predicted = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    y_predicted = y_predicted.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, y_predicted, alpha=0.8, cmap='viridis')
    scatter_train = plt.scatter(X_train_s[:, 0], X_train_s[:, 1], c=y_train, cmap='viridis', edgecolors='k', label='Training data')
    scatter_test = plt.scatter(X_test_s[:, 0], X_test_s[:, 1], c=y_test, cmap='viridis', edgecolors='w', linewidth=1, label='Test data')
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title('SVC Decision Boundary')
    cbar = plt.colorbar(scatter_train)
    cbar.set_ticks(list(label_mapping.values()))
    cbar.set_ticklabels(list(label_mapping.keys()))
    cbar.set_label('Typ')
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    
if graph_flags['feature_space']:
    if X.shape[1] >= 2:
        X_vis = X.iloc[:, :2]
        X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
            X_vis, y, test_size=0.2, random_state=9, stratify=y
        )
        scaler_vis = StandardScaler()
        X_train_vis_scaled = scaler_vis.fit_transform(X_train_vis)
        X_test_vis_scaled = scaler_vis.transform(X_test_vis)
        # MLP
        mlp_vis = models[0][1]
        mlp_vis.fit(X_train_vis_scaled, y_train_vis)
        plot_decision_boundary(mlp_vis, X_train_vis_scaled, y_train_vis.values, X_test_vis_scaled, y_test_vis.values,
                            "MLP Decision Boundary", label_mapping)
        # KNN
        knn_vis = models[1][1]
        knn_vis.fit(X_train_vis_scaled, y_train_vis)
        plot_decision_boundary(knn_vis, X_train_vis_scaled, y_train_vis.values, X_test_vis_scaled, y_test_vis.values,
                            "KNN Decision Boundary", label_mapping)
        # SVC
        svc_vis = models[2][1]
        plot_svc_decision_boundary(svc_vis, X_train_vis_scaled, y_train_vis.values, X_test_vis_scaled, y_test_vis.values, label_mapping)
        # Decision Tree
        dt_vis = models[3][1]
        dt_vis.fit(X_train_vis_scaled, y_train_vis)
        plot_decision_tree(dt_vis, X_vis.columns, list(label_mapping.keys()), "Decision Tree Visualization")
        # Random Forest
        rf_vis = models[4][1]
        rf_vis.fit(X_train_vis_scaled, y_train_vis)
        plot_random_forest_tree(rf_vis, X_vis.columns, list(label_mapping.keys()), "Visualization of One Tree from the Random Forest")
    else:
        print("Not enough features for 2D decision boundary plots.")

    plt.show()
    
if graph_flags['cv_summary']:
    plt.figure(figsize=(8, 5))
    for model, _ in models:
        scores = []
        for split in split_counts:
            for name, score in all_rankings[split]:
                if name == model:
                    scores.append(score)
        plt.plot(split_counts, scores, marker='o', label=model)
    plt.title("Mean CV Accuracy vs Split Count")
    plt.xlabel("CV Split Count")
    plt.ylabel("Mean CV Accuracy")
    plt.xticks(split_counts)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

    for cls in label_mapping.keys():
        plt.figure(figsize=(8, 5))
        for model, _ in models:
            recalls = []
            for split in split_counts:
                recalls.append(all_class_acc[split][model][cls])
            plt.plot(split_counts, recalls, marker='o', label=model)
        plt.title(f"Per-class Mean Recall for '{cls}'")
        plt.xlabel("CV Split Count")
        plt.ylabel("Mean Recall")
        plt.xticks(split_counts)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

    # summary table
    print("\n" + "="*60)
    print("Mean CV accuracy vs split count for each model:")
    print("="*60)
    print(f"{'Model':<15} {'CV=5':>8} {'CV=6':>8} {'CV=7':>8}")
    for model, _ in models:
        scores = []
        for split in split_counts:
            for name, score in all_rankings[split]:
                if name == model:
                    scores.append(f"{score:.3f}")
        print(f"{model:<15} {scores[0]:>8} {scores[1]:>8} {scores[2]:>8}")
    print("="*60)

    # summary per-class table
    print("\n" + "="*60)
    print("Per-class mean precision for each model and split count:")
    print("="*60)
    header = f"{'Model':<15}" + "".join([f"{cls:>12}" for cls in label_mapping.keys()])
    print(header)
    for model, _ in models:
        for split in split_counts:
            line = f"{model:<15} (CV={split})"
            for cls in label_mapping.keys():
                precision = all_class_acc[split][model][cls]
                line += f"{precision:>12.3f}"
            print(line)
    print("="*60)

    plt.show()