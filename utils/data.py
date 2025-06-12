
import pandas as pd
from matplotlib import pyplot as plt

from utils.plots import plot_data_scatter

label_mapping = {'light': 0, 'medium': 1, 'heavy': 2}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

def load_data(filename, graph_flags):
    df = pd.read_csv(filename)
    print("\n=== FULL DATASET ===")
    print(df.to_string(index=False))
    print("====================\n")
    if 'Nr' in df.columns:
        df = df.drop('Nr', axis='columns')
    df['Typ'] = df['Typ'].map(label_mapping)
    X = df.drop(columns=['ID próbki', 'Typ', 'S (%)', 'Ar (%)', 'R (%)', 'As (%)'])
    y = df['Typ']

    if graph_flags['data_scatter']:
        plot_data_scatter(df)
    plt.show()

    return X, y, df