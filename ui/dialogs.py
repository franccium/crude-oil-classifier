import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import ui.state as state
from ui.utils import ranking
import os
import threading


def select_dataset():
    def set_choice(choice):
        state.selected_file = os.path.join("./data", choice)
        dataset_window.destroy()

    dataset_window = tk.Toplevel(state.root)
    dataset_window.title("Choose Dataset")
    dataset_window.geometry("300x200")
    dataset_window.grid_columnconfigure(0, weight=1)

    tk.Label(dataset_window, text="Choose dataset:").grid(row=0, column=0, pady=20, padx=10, sticky="n")

    tk.Button(dataset_window, text="Original", command=lambda: set_choice("data_original.csv")).grid(row=1, column=0,
                                                                                                     pady=15, padx=20,
                                                                                                     sticky="ew")
    tk.Button(dataset_window, text="Augmented", command=lambda: set_choice("data_augmented.csv")).grid(row=2, column=0,
                                                                                                       pady=15, padx=20,
                                                                                                       sticky="ew")

def select_graphs():
    def submit():
        state.graph_flags.update({k: v.get() for k, v in flags.items()})
        graphs_window.destroy()

    graphs_window = tk.Toplevel(state.root)
    graphs_window.title("Select Graphs to Show")
    graphs_window.geometry("350x250")
    graphs_window.grid_columnconfigure(0, weight=1)

    flags = {
        'feature_space': tk.BooleanVar(value=True),
        'data_scatter': tk.BooleanVar(value=True),
        'cv_summary': tk.BooleanVar(value=True)
    }

    tk.Label(graphs_window, text="Select which graphs to display:").grid(row=0, column=0, pady=10, sticky="w")

    tk.Checkbutton(graphs_window, text="Show feature space visualizations", variable=flags['feature_space']).grid(row=1,
                                                                                                                  column=0,
                                                                                                                  sticky="w",
                                                                                                                  padx=10)
    tk.Checkbutton(graphs_window, text="Show data scatter plots", variable=flags['data_scatter']).grid(row=2, column=0,
                                                                                                       sticky="w",
                                                                                                       padx=10)
    tk.Checkbutton(graphs_window, text="Show cross-validation summary graphs", variable=flags['cv_summary']).grid(row=3,
                                                                                                                  column=0,
                                                                                                                  sticky="w",
                                                                                                                  padx=10)

    tk.Button(graphs_window, text="OK", command=submit).grid(row=4, column=0, pady=15)


def show_loading(message="Please wait..."):
    loading = tk.Toplevel(state.root)
    loading.title("Loading...")
    loading.geometry("250x100")
    loading.grab_set()
    tk.Label(loading, text=message).pack(pady=20)

    progress = ttk.Progressbar(loading, mode='indeterminate')
    progress.pack(fill='x', padx=20, pady=5)
    progress.start()

    return loading

def select_featureset():

    def set_choice(choice):
        state.featureset = choice
        dialog.destroy()

    dialog = tk.Toplevel(state.root)
    dialog.title("Choose Feature Set")
    dialog.geometry("300x200")
    dialog.grab_set()
    dialog.transient(state.root)
    dialog.grid_columnconfigure(0, weight=1)

    tk.Label(dialog, text="Choose feature set:").grid(row=0, column=0, padx=20, pady=15)
    tk.Button(dialog, text="Density & CII", width=25, command=lambda: set_choice(state.featureset_density_CII)).grid(row=1, column=0, pady=5, padx=30)
    tk.Button(dialog, text="Density Group", width=25, command=lambda: set_choice(state.featureset_density_group)).grid(row=2, column=0, pady=5, padx=30)
    tk.Button(dialog, text="Density Group + CII", width=25, command=lambda: set_choice(state.featureset_density_group_CII)).grid(row=3, column=0, pady=5, padx=30)

    dialog.wait_window()
    return

def get_best_model_for_featureset():
    featureset = state.featureset

    model_map = {
        (0, 'original'): 'mlp_density_cii.pkl',
        (0, 'augmented'): 'decision tree_aug_density_CII.pkl',
        (1, 'original'): 'random forest_density_group.pkl',
        (1, 'augmented'): 'knn_aug_density_group.pkl',
        (2, 'original'): 'decision tree_density_group_cii.pkl',
        (2, 'augmented'): 'mlp_aug_density_group_cii.pkl',
    }

    data_type = {'value': None}

    def set_data_type(choice):
        data_type['value'] = choice
        dialog.destroy()

    dialog = tk.Toplevel(state.root)
    dialog.title("Choose Data Type")
    dialog.geometry("300x150")
    dialog.grab_set()
    dialog.transient(state.root)
    dialog.grid_columnconfigure(0, weight=1)

    tk.Label(dialog, text="Use model trained on:").grid(row=0, column=0, padx=20, pady=15)
    tk.Button(dialog, text="Original Data", width=25, command=lambda: set_data_type('original')).grid(row=1, column=0, pady=5, padx=30)
    tk.Button(dialog, text="Augmented Data", width=25, command=lambda: set_data_type('augmented')).grid(row=2, column=0, pady=5, padx=30)

    dialog.wait_window()

    if data_type['value'] is None:
        return None

    featureset_id = state.reverse_featureset_map.get(tuple(featureset))
    if featureset_id is None:
        return None

    best_model = model_map.get((featureset_id, data_type['value']))
    state.best_model = best_model

def show_ranking_dialog():
    dialog = tk.Toplevel(state.root)
    dialog.title("Ranking")
    dialog.geometry("250x120")
    dialog.grab_set()
    dialog.transient(state.root)
    dialog.grid_columnconfigure(0, weight=1)

    label = tk.Label(dialog, text="Loading...")
    label.grid(row=0, column=0, padx=30, pady=25)

    def run_ranking():
        ranking()
        dialog.after(0, lambda: label.config(text="Details in console."))

    threading.Thread(target=run_ranking, daemon=True).start()

    tk.Button(dialog, text="OK", command=dialog.destroy).grid(row=1, column=0, pady=10)