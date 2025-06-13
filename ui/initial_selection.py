
import tkinter as tk
import os

featureset_density_CII = ['Density', 'CII']
featureset_density_group = ['Density', 'S', 'Ar', 'R', 'As']
featureset_density_group_CII = ['Density', 'S', 'Ar', 'R', 'As', 'CII']
featureset_map = {0: featureset_density_CII, 1: featureset_density_group, 2: featureset_density_group_CII}
reverse_featureset_map = {tuple(v): k for k, v in featureset_map.items()}

def select_featureset():
    def set_choice(choice):
        nonlocal selected_featureset
        selected_featureset = choice
        root.destroy()

    selected_featureset = None
    root = tk.Tk()
    root.title("Choose feature set")
    tk.Label(root, text="Choose feature set:").pack(padx=20, pady=10)
    tk.Button(root, text="Density & CII", width=25, command=lambda: set_choice(featureset_density_CII)).pack(pady=5)
    tk.Button(root, text="Density Group", width=25, command=lambda: set_choice(featureset_density_group)).pack(pady=5)
    tk.Button(root, text="Density Group + CII", width=25, command=lambda: set_choice(featureset_density_group_CII)).pack(pady=5)
    root.mainloop()
    return selected_featureset

def select_mode():
    selected_mode = None
    def set_mode(mode):
        nonlocal selected_mode
        selected_mode = mode
        root.destroy()
    root = tk.Tk()
    root.title("Choose Mode")
    tk.Label(root, text="Choose what to do:").pack(padx=20, pady=10)
    tk.Button(root, text="Train Models", width=25, command=lambda: set_mode("train")).pack(pady=5)
    tk.Button(root, text="Test Exported Model", width=25, command=lambda: set_mode("test")).pack(pady=5)
    root.mainloop()
    return selected_mode

def select_model_file():
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select exported model",
        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path

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

def get_best_model_for_featureset(featureset):
    model_map = {
        (0, 'original'): 'mlp_density_cii.pkl',
        (0, 'augmented'): 'decision tree_aug_density_CII.pkl',
        (1, 'original'): 'random forest_density_group.pkl',
        (1, 'augmented'): 'knn_aug_density_group.pkl',
        (2, 'original'): 'decision tree_density_group_cii.pkl',
        (2, 'augmented'): 'mlp_aug_density_group_cii.pkl',
    }

    data_type = None
    def set_data_type(choice):
        nonlocal data_type
        data_type = choice
        root2.destroy()

    root2 = tk.Tk()
    root2.title("Choose Data Type")
    tk.Label(root2, text="Use model trained on:").pack(padx=20, pady=10)
    tk.Button(root2, text="Original Data", width=25, command=lambda: set_data_type('original')).pack(pady=5)
    tk.Button(root2, text="Augmented Data", width=25, command=lambda: set_data_type('augmented')).pack(pady=5)
    root2.mainloop()
    if data_type is None:
        return None
    
    featureset_id = reverse_featureset_map.get(tuple(featureset))
    best_model = model_map[(featureset_id, data_type)]
    print("Best model found: " + best_model)

    return best_model