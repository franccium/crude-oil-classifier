
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report
import pandas as pd

from utils.plots import plot_cv_results

import joblib
from datetime import datetime
import os
    
def find_and_export_best_model(X, y, all_rankings, models):
    best_split = None
    best_model_name = None
    best_score = -1

    for split, ranking in all_rankings.items():
        model_name, score = ranking[0]
        if score > best_score:
            best_score = score
            best_split = split
            best_model_name = model_name

    print(f"Best split: {best_split}")
    print(f"Best model: {best_model_name}")
    print(f"Best mean CV accuracy: {best_score:.3f}")

    best_model = None
    for name, model in models:
        if name == best_model_name:
            best_model = model
            break

    best_model.fit(X, y)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{best_model_name.lower()}_{timestamp}.pkl"
    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)
    filepath = os.path.join(export_dir, filename)
    joblib.dump(best_model, filepath)
    print(f"Exported best model to {filepath}")

def evaluate_models_cv(models, X, y, split_counts,
                       label_mapping, reverse_label_mapping, export_best_model = False):
    all_rankings = {}
    all_class_acc = {}

    for split in split_counts:
        print("\n" + "=" * 60)
        print(f" CROSS-VALIDATION SPLIT COUNT: {split} ".center(60, "="))
        print("=" * 60)
        model_scores = []
        class_acc = {}

        cv = StratifiedKFold(n_splits=split, shuffle=True, random_state=9)

        for name, model in models:
            print("=" * 50)
            print(name)
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            mean_acc = cv_scores.mean()
            model_scores.append((name, mean_acc))

            y_pred_cv = cross_val_predict(model, X, y, cv=cv)
            y_str = y.map(reverse_label_mapping)
            y_pred_cv_str = pd.Series(y_pred_cv).map(reverse_label_mapping)

            report = classification_report(y_str, y_pred_cv_str, target_names=label_mapping.keys(), output_dict=True)
            print(classification_report(y_str, y_pred_cv_str, target_names=label_mapping.keys()))

            class_acc[name] = {cls: report[cls]['recall'] for cls in label_mapping.keys()}
            print(f"Mean CV accuracy: {mean_acc:.3f}")

        # Overall ranking
        print("\n" + "=" * 50)
        print(f"FINAL MODEL RANKING (CV splits: {split})")
        print("=" * 50)
        ranking = sorted(model_scores, key=lambda x: x[1], reverse=True)
        all_rankings[split] = ranking
        all_class_acc[split] = class_acc

        for i, (name, score) in enumerate(ranking, 1):
            print(f"{i}. {name:<15} {score:.3f}")
        print("=" * 50)

        # Per-class ranking
        print("\nPer-class accuracy ranking:")
        for cls in label_mapping.keys():
            print(f"\nClass: {cls}")
            class_ranking = sorted(
                [(name, class_acc[name][cls]) for name in class_acc],
                key=lambda x: x[1], reverse=True
            )
            for i, (name, score) in enumerate(class_ranking, 1):
                print(f"{i}. {name:<15} {score:.3f}")
                
    if export_best_model:
        find_and_export_best_model(X, y, all_rankings, models)

    return all_rankings, all_class_acc

def report_cv_results(models, split_counts, all_rankings, all_class_acc, label_mapping):
    print("\n" + "=" * 60)
    print("Mean CV accuracy vs split count for each model:")
    print("=" * 60)
    header = f"{'Model':<15}" + "".join([f"{f'CV={split}':>8}" for split in split_counts])
    print(header)
    for model_name, _ in models:
        scores = []
        for split in split_counts:
            for name, score in all_rankings[split]:
                if name == model_name:
                    scores.append(f"{score:.3f}")
        print(f"{model_name:<15}" + "".join([f"{score:>8}" for score in scores]))
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Per-class mean precision for each model and split count:")
    print("=" * 60)
    header = f"{'Model':<15}" + "".join([f"{cls:>12}" for cls in label_mapping.keys()])
    print(header)
    for model_name, _ in models:
        for split in split_counts:
            line = f"{model_name:<15} (CV={split})"
            for cls in label_mapping.keys():
                precision = all_class_acc[split][model_name][cls]
                line += f"{precision:>12.3f}"
            print(line)
    print("=" * 60)
    plot_cv_results(models, split_counts, all_rankings, all_class_acc, label_mapping)