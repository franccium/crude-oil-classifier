import pandas as pd
import numpy as np



def gen_aug():
    data = pd.read_csv('data.csv')
    num_cols = ['Gęstość', 'S (%)', 'Ar (%)', 'R (%)', 'As (%)', 'CII']
    noise_level = 0.02
    n_aug_per_category = 10
    augmented_rows = []
    current_max_nr = data['Nr'].max()
    categories = data['Typ'].unique()

    for cat in categories:
        cat_data = data[data['Typ'] == cat]

        for i in range(n_aug_per_category):
            row = cat_data.sample(n=1).iloc[0].copy()
            for col in num_cols:
                value = row[col]
                noise = np.random.normal(0, noise_level * value)
                row[col] = max(value + noise, 0)

            current_max_nr += 1
            row['Nr'] = current_max_nr
            row['ID próbki'] = f"{row['ID próbki']}_aug_{cat}_{i+1}"
            augmented_rows.append(row)

    augmented_data = pd.DataFrame(augmented_rows)
    combined_data = pd.concat([data, augmented_data], ignore_index=True)
    combined_data.to_csv('augmented_data_balanced.csv', index=False)

def sara_aug(df, max_aug=15):
    columns_to_augment = ["S", "Ar", "R", "As", "CII"]
    n_augmentations = 3
    noise_level = 0.05

    augmented_rows = []
    for _, row in df.iterrows():
        for _ in range(n_augmentations):
            noisy_values = row[columns_to_augment] * (1 + np.random.normal(0, noise_level, len(columns_to_augment)))
            augmented_rows.append(noisy_values.tolist())

    augmented_df = pd.DataFrame(augmented_rows, columns=columns_to_augment)

    if len(augmented_df) > max_aug:
        augmented_df = augmented_df.sample(n=20).reset_index(drop=True)

    return augmented_df

def augment_data(df, columns_to_augment, n_augmentations, max_aug=15, noise_level=0.05):
    augmented_rows = []
    for _, row in df.iterrows():
        for _ in range(n_augmentations):
            noisy_values = row[columns_to_augment] * (1 + np.random.normal(0, noise_level, len(columns_to_augment)))
            augmented_rows.append(noisy_values.tolist())

    augmented_df = pd.DataFrame(augmented_rows, columns=columns_to_augment)

    if len(augmented_df) > max_aug:
        augmented_df = augmented_df.sample(n=20).reset_index(drop=True)

    return augmented_df