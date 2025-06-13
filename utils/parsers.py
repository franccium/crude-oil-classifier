
import os
import re
import pandas as pd

def parse_sara(filename='sara.csv'):
    file = os.path.join("..", "data", filename)
    df = pd.read_csv(file)

    def parse_composition(comp):
        match = re.match(r"(\d+)(?=%[A-Z])(?!\d)%([A-Z]+)/(\d+)(?=%[A-Z])(?!\d)%([A-Z]+)", comp)
        if not match:
            raise ValueError("Input csv wrongly defined")
        pct1, id1, pct2, id2 = match.groups()
        return [id1, int(pct1), id2, int(pct2)]

    parsed = df['Composition'].apply(parse_composition)
    parsed_df = pd.DataFrame(parsed.tolist(), columns=[
        'ID_1', '%_1', 'ID_2', '%_2'
    ])

    final_df = pd.concat([
        df[['Nr']],
        parsed_df,
        df.drop(columns=['Nr', 'Composition'])
    ], axis=1)
    return final_df

def parse_s_value(filename='s_value.csv'):
    file = os.path.join("..", "data", filename)
    df = pd.read_csv(file)

    result_df = df.drop(columns=['ID_1', 'ID_2'])

    return result_df
