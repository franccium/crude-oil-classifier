import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from ui.dialogs import select_dataset, select_graphs, show_loading, select_featureset, get_best_model_for_featureset, show_ranking_dialog
from ui.utils import clear_window, load_sample_ids
import ui.state as state
from ui.Mix import Mix
import time
import threading

def show_main_screen():
    clear_window(state.root)
    state.root.title("Oil Classifier App")
    state.root.grid_rowconfigure(0, weight=1)
    state.root.grid_rowconfigure(1, weight=1)
    state.root.grid_columnconfigure(0, weight=1)
    state.root.grid_columnconfigure(1, weight=0)

    tk.Button(state.root, text="Model ranking", width=20, command=show_ranking_screen).grid(
        row=0, column=0, pady=70, padx=100, sticky="nsew"
    )
    tk.Button(state.root, text="Samples mixing", width=20, command=show_sample_selection).grid(
        row=1, column=0, pady=70, padx=100, sticky="nsew"
    )

def show_ranking_screen():
    print("here")
    clear_window(state.root)
    state.root.title("Oil Classifier Ranking")

    for i in range(6):
        state.root.grid_rowconfigure(i, weight=1)
    state.root.grid_columnconfigure(0, weight=1)
    state.root.grid_columnconfigure(1, weight=1)

    tk.Label(state.root, text="Welcome to the Oil Classifier", font=("Helvetica", 16)).grid(
        row=0, column=0, columnspan=2, pady=20, sticky="n"
    )

    tk.Button(state.root, text="Select Dataset", width=20, command=select_dataset).grid(
        row=1, column=0, columnspan=2, pady=5, padx=60, sticky="nsew"
    )
    tk.Button(state.root, text="Select Graphs Options", width=20, command=select_graphs).grid(
        row=2, column=0, columnspan=2, pady=5, padx=60,sticky="nsew"
    )
    tk.Button(state.root, text="Select featureset", width=20, command=select_featureset).grid(
        row=3, column=0, columnspan=2, pady=5, padx=60,sticky="nsew"
    )
    tk.Button(state.root, text="Select model", width=20, command=get_best_model_for_featureset).grid(
        row=4, column=0, columnspan=2, pady=5, padx=60,sticky="nsew"
    )

    tk.Button(state.root, text="Back", width=20, command=show_main_screen).grid(
        row=5, column=0, pady=20, padx=60,sticky="nsew"
    )
    tk.Button(state.root, text="Next", width=20, command=go_next_ranking).grid(
        row=5, column=1, pady=20, padx=60,sticky="nsew"
    )

def show_sample_selection():
    sample_ids = load_sample_ids(state.selected_file)

    clear_window(state.root)
    state.root.title("Select Samples")

    main_frame = tk.Frame(state.root)
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

    tk.Button(state.root, text="Back", width=20, command=lambda: go_back()).grid(row=4, column=0, sticky="sw", pady=30, padx=50)
    tk.Button(state.root, text="Confirm", width=20, command=lambda: confirm()).grid(row=4, column=0, sticky="se", pady=30, padx=50)

    def confirm():
        s1, s2 = sample1.get(), sample2.get()
        v1, v2 = value1.get(), value2.get()

        if not s1 or not s2:
            messagebox.showwarning("Missing samples", "Please select both samples.")
            return

        loading = show_loading(state.root)

        def background_task():
            try:
                mix_result = Mix(s1, s2, v1, v2, state.selected_file)
                # after creating mix, we can call functions from mix which will predict values
                print("Selected graphs:", state.graph_flags)
                print(
                    f"Selected mix from samples: {mix_result.id1} and {mix_result.id2}. In ratio: {mix_result.v1}:{mix_result.v2}.")
                print(f"Sara of sample 1: {mix_result.id1}: {mix_result.sara1}. ")
                print(f"Sara of sample 2: {mix_result.id2}: {mix_result.sara2}. ")
                print(f"Best model for featureset: {state.best_model}")

                def on_done():
                    state.mix = mix_result
                    loading.destroy()
                    show_result_screen()

                state.root.after(0, on_done)

            except Exception as e:
                def show_error(e=e):
                    loading.destroy()
                    messagebox.showerror("Error", f"Failed to create mix:\n{e}")

                state.root.after(0, show_error)

        threading.Thread(target=background_task).start()

    def go_back():
        show_main_screen()

def go_next_ranking():
    if not state.selected_file or not state.featureset or not state.best_model:
        messagebox.showwarning("Missing option", "Please select from all options before proceeding.")
        return
    show_ranking_dialog()

def show_result_screen():
    clear_window(state.root)
    state.root.title("Results")
    state.root.configure(bg="#f5f5f5")

    for i in range(4):
        state.root.grid_rowconfigure(i, weight=1)
    state.root.grid_columnconfigure(0, weight=1)


    mix = state.mix

    type_color = {
        "light": "green",
        "medium": "orange",
        "heavy": "red"
    }

    header_frame = ttk.Frame(state.root, padding=20)
    header_frame.grid(row=0, column=0, sticky="ew")
    header_frame.columnconfigure((0, 1), weight=1)

    ttk.Label(header_frame, text=f"Sample ID1: {mix.id1} ({mix.v1}%)",
              font=("Helvetica", 12)).grid(row=0, column=0, sticky="e", padx=10)
    ttk.Label(header_frame, text=f"Sample ID2: {mix.id2} ({mix.v2}%)",
              font=("Helvetica", 12)).grid(row=0, column=1, sticky="w", padx=10)

    ttk.Label(header_frame, text=mix.type1, font=("Helvetica", 12, "bold"),
              foreground=type_color.get(mix.type1, "black")
    ).grid(row=2, column=0, sticky="e", padx=10)

    ttk.Label(header_frame, text=mix.type2, font=("Helvetica", 12, "bold"),
        foreground=type_color.get(mix.type2, "black")
    ).grid(row=2, column=1, sticky="w", padx=10)


    ttk.Label(state.root, text=f"Predicted Stability: {mix.predicted}",
              font=("Helvetica", 13, "bold"), foreground="#2b7a0b").grid(row=1, column=0, pady=(10, 20))


    headers = ["CII", "S-Value", "P-Value", "TSI"]
    values = [str(mix.CII), str(mix.Svalue), str(mix.Pvalue), str(mix.TSI)]

    table_frame = tk.Frame(state.root, bg="white", highlightbackground="#cccccc", highlightthickness=1)
    table_frame.grid(row=2, column=0, pady=10, padx=20)

    for col in range(len(headers)):
        table_frame.grid_columnconfigure(col, weight=1, minsize=100)

    for col, text in enumerate(headers):
        header_cell = tk.Frame(table_frame, bg="#e6e6e6", padx=10, pady=8)
        header_cell.grid(row=0, column=col, sticky="nsew", padx=1, pady=1)
        tk.Label(header_cell, text=text, font=("Helvetica", 11, "bold"), bg="#e6e6e6").pack(fill="both", expand=True)

    for col, value in enumerate(values):
        value_cell = tk.Frame(table_frame, bg="white", padx=10, pady=8)
        value_cell.grid(row=1, column=col, sticky="nsew", padx=1, pady=1)
        tk.Label(value_cell, text=value, font=("Helvetica", 11), bg="white").pack(fill="both", expand=True)

    ttk.Button(state.root, text="Back", command=show_main_screen).grid(row=3, column=0, pady=20)
