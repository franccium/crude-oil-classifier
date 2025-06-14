import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import state
import os


def select_dataset():
    def set_choice(choice):
        state.selected_file = os.path.join("../data", choice)
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
