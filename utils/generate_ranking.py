from collections import defaultdict

import pandas as pd
import os

stab_data = pd.read_csv(os.path.join('..', 'data', 'joined.csv'))
label_mapping = {'lekkie': 0, 'średnie': 1, 'ciężkie': 2}


stab_data['Typ'] = stab_data['Typ'].map(label_mapping)
stab_data.head()

method_names = {
    1: 'TURBISCAN',
    2: 'SARA',
    3: 'S-VALUE',
    4: 'P-VALUE',
    5: 'Testy bibułowe'
}

stab_data['Metody skuteczne'] = stab_data['Metody skuteczne w określeniu stabilności'].astype(str)\
    .str.replace('"', '').str.strip().str.split(',')\
    .apply(lambda x: [int(i) for i in x if i.strip().isdigit()])

method_counts = {0: defaultdict(int), 1: defaultdict(int), 2: defaultdict(int)}

for _, row in stab_data.iterrows():
    oil_type = row['Typ']
    for method in row['Metody skuteczne']:
        method_counts[oil_type][method] += 1

for oil_type in method_counts:
    total = len(stab_data[stab_data['Typ'] == oil_type])
    for method in method_counts[oil_type]:
        method_counts[oil_type][method] = (method_counts[oil_type][method] / total) * 100

rankings = {}
for oil_type in method_counts:
    sorted_methods = sorted(method_counts[oil_type].items(), key=lambda x: (-x[1], x[0]))
    rankings[oil_type] = [(method_names[method], f"{percentage:.1f}%") for method, percentage in sorted_methods]

reverse_label_mapping = {0: 'lekkie', 1: 'średnie', 2: 'ciężkie'}

def display_method_ranking(oil_type_index):
    print(f"\nRanking for {reverse_label_mapping[oil_type_index]} crude oil:")
    print("{:<20} {:<15}".format("Method", "Effectiveness"))
    print("-" * 35)
    for method, percentage in rankings[oil_type_index]:
        print("{:<20} {:<15}".format(method, percentage))

# all rankings
print("Effectiveness Ranking of Stability Assessment Methods by Crude Oil Type")
print("=" * 70)
for oil_type in rankings:
    display_method_ranking(oil_type)