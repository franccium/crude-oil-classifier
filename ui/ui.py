import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import os
from ui_tools import load_sample_ids
from Mix import Mix

selected_file = None
graph_flags = {
    'feature_space': True,
    'data_scatter': True,
    'cv_summary': True
}
mix = None

def select_dataset(parent):
    def set_choice(choice):
        global selected_file
        selected_file = os.path.join("../data", choice)
        dataset_window.destroy()

    dataset_window = tk.Toplevel(parent)
    dataset_window.title("Choose Dataset")
    dataset_window.geometry("300x200")
    dataset_window.grid_columnconfigure(0, weight=1)

    tk.Label(dataset_window, text="Choose dataset:").grid(row=0, column=0, pady=20, padx=10, sticky="n")

    tk.Button(dataset_window, text="Original", command=lambda: set_choice("data_original.csv")).grid(row=1, column=0, pady=15, padx=20, sticky="ew")
    tk.Button(dataset_window, text="Augmented", command=lambda: set_choice("data_augmented.csv")).grid(row=2, column=0, pady=15, padx=20, sticky="ew")


def select_graphs(parent):
    def submit():
        global graph_flags
        graph_flags.update({k: v.get() for k, v in flags.items()})
        graphs_window.destroy()

    graphs_window = tk.Toplevel(parent)
    graphs_window.title("Select Graphs to Show")
    graphs_window.geometry("350x250")
    graphs_window.grid_columnconfigure(0, weight=1)

    flags = {
        'feature_space': tk.BooleanVar(value=True),
        'data_scatter': tk.BooleanVar(value=True),
        'cv_summary': tk.BooleanVar(value=True)
    }

    tk.Label(graphs_window, text="Select which graphs to display:").grid(row=0, column=0, pady=10, sticky="w")

    tk.Checkbutton(graphs_window, text="Show feature space visualizations", variable=flags['feature_space']).grid(row=1, column=0, sticky="w", padx=10)
    tk.Checkbutton(graphs_window, text="Show data scatter plots", variable=flags['data_scatter']).grid(row=2, column=0, sticky="w", padx=10)
    tk.Checkbutton(graphs_window, text="Show cross-validation summary graphs", variable=flags['cv_summary']).grid(row=3, column=0, sticky="w", padx=10)

    tk.Button(graphs_window, text="OK", command=submit).grid(row=4, column=0, pady=15)


def clear_window(window):
    for widget in window.winfo_children():
        widget.destroy()


def show_main_screen():
    clear_window(root)
    root.title("Oil Classifier App")

    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(2, weight=1)
    root.grid_rowconfigure(3, weight=3)
    root.grid_columnconfigure(0, weight=1)

    tk.Label(root, text="Welcome to the Oil Classifier", font=("Helvetica", 16)).grid(row=0, column=0, pady=20, sticky="n")

    tk.Button(root, text="Select Dataset", width=20, command=lambda: select_dataset(root)).grid(row=1, column=0, pady=5, sticky="n")
    tk.Button(root, text="Select Graphs Options", width=20, command=lambda: select_graphs(root)).grid(row=2, column=0, pady=5, sticky="n")
    tk.Button(root, text="Next", width=20, command=go_next).grid(row=3, column=0, pady=20, sticky="n")


def show_sample_selection():
    sample_ids = load_sample_ids(selected_file)

    clear_window(root)
    root.title("Select Samples")

    main_frame = tk.Frame(root)
    main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
    for i in range(2):
        main_frame.grid_columnconfigure(i, weight=1)

    value1 = tk.IntVar(value=50)
    value2 = tk.IntVar(value=50)

    def on_scale1_change(val):
        val = int(float(val))
        value1.set(val)
        value2.set(100 - val)
        scale2.set(100 - val)

    def on_scale2_change(val):
        val = int(float(val))
        value2.set(val)
        value1.set(100 - val)
        scale1.set(100 - val)

    tk.Label(main_frame, text="Select Sample 1:").grid(row=0, column=0, sticky="w", padx=50)
    sample1 = ttk.Combobox(main_frame, values=sample_ids)
    sample1.grid(row=1, column=0, sticky="ew", pady=(0, 10), padx=50)

    tk.Label(main_frame, text="Select Sample 2:").grid(row=0, column=1, sticky="w", padx=50)
    sample2 = ttk.Combobox(main_frame, values=sample_ids)
    sample2.grid(row=1, column=1, sticky="ew", pady=(0, 10), padx=50)

    tk.Label(main_frame, text="Percentage of the Sample 1:").grid(row=2, column=0, sticky="w", pady=(10, 0), padx=50)
    tk.Label(main_frame, text="Percentage of the Sample 2:").grid(row=2, column=1, sticky="w", pady=(10, 0), padx=50)

    scale1 = tk.Scale(main_frame, from_=0, to=100, orient='horizontal', variable=value1, command=on_scale1_change)
    scale1.grid(row=3, column=0, sticky="ew", pady=10, padx=50)

    scale2 = tk.Scale(main_frame, from_=0, to=100, orient='horizontal', variable=value2, command=on_scale2_change)
    scale2.grid(row=3, column=1, sticky="ew", pady=10, padx=50)

    def confirm():
        s1, s2 = sample1.get(), sample2.get()
        v1, v2 = value1.get(), value2.get()

        if not s1 or not s2:
            messagebox.showwarning("Missing samples", "Please select both samples.")
        else:
            print("Selected graphs:", graph_flags)
            mix = Mix(s1, s2, v1, v2, selected_file)
            print(f"Selected mix from samples: {mix.id1} and {mix.id2}. In ratio: {mix.v1}:{mix.v2}.")
            print(f"Sara of sample 1: {mix.id1}: {mix.sara1}. ")
            print(f"Sara of sample 2: {mix.id2}: {mix.sara2}. ")
            show_main_screen()

    def go_back():
        show_main_screen()

    tk.Button(root, text="Back", width=20, command=go_back).grid(row=4, column=0, sticky="sw", pady=30, padx=50)
    tk.Button(root, text="Confirm", width=20, command=confirm).grid(row=4, column=0, sticky="se", pady=30, padx=50)


def go_next():
    if not selected_file:
        messagebox.showwarning("Missing dataset", "Please select a dataset first.")
        return
    show_sample_selection()

root = tk.Tk()
root.title("Oil Classifier")
root.geometry("600x400")
root.minsize(600, 400)

show_main_screen()
root.mainloop()