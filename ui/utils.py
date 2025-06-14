import csv

def load_sample_ids(selected_file):
    sample_ids = []
    with open(selected_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            if row:
                sample_ids.append(row[0])
    return sorted(set(sample_ids))

def clear_window(window):
    for widget in window.winfo_children():
        widget.destroy()
