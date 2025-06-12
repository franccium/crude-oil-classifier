
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree

def plot_decision_boundary(clf, X_train, y_train, X_test, y_test,
                           label_mapping, x_label, y_label, title):

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
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
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

def plot_svc_decision_boundary(clf, X_train, y_train, X_test, y_test, label_mapping,
                               x_label, y_label):
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
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('SVC Decision Boundary')
    cbar = plt.colorbar(scatter_train)
    cbar.set_ticks(list(label_mapping.values()))
    cbar.set_ticklabels(list(label_mapping.keys()))
    cbar.set_label('Typ')
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

def plot_data_scatter(df):
    custom_palette = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}
    label_names = {0: 'light', 1: 'medium', 2: 'heavy'}
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

def plot_cv_results(models, split_counts, all_rankings, all_class_acc, label_mapping):
    n_classes = len(label_mapping)
    n_subplots = 1 + n_classes
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Plot overall accuracy
    ax = axes[0]
    for model_name, _ in models:
        scores = [score for split in split_counts
                  for name, score in all_rankings[split]
                  if name == model_name]
        ax.plot(split_counts, scores, marker='o', label=model_name)
    ax.set_title("Mean CV Accuracy")
    ax.set_xlabel("CV Split Count")
    ax.set_ylabel("Mean CV Accuracy")
    ax.set_xticks(split_counts)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)

    # Plot per-class precision
    for idx, cls in enumerate(label_mapping.keys()):
        ax = axes[idx + 1]
        for model_name, _ in models:
            precisions = [all_class_acc[split][model_name][cls] for split in split_counts]
            ax.plot(split_counts, precisions, marker='o', label=model_name)
        ax.set_title(f"Precision: {cls}")
        ax.set_xlabel("CV Split Count")
        ax.set_ylabel("Mean Precision")
        ax.set_xticks(split_counts)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True)

    # Remove unused subplots
    for i in range(n_subplots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()