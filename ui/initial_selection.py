
import tkinter as tk
import os

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
    return os.path.join("data", selected_file)

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
