
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

def parse_s_value(filename='s_value_with_As.csv'):
    file = os.path.join("..", "data", filename)
    df = pd.read_csv(file)
    df['S_Value_part1'] = df['S_Value1'] * (df['%_1'] / 100)
    df['S_Value_part2'] = df['S_Value2'] * (df['%_2'] / 100)

    result_df = df[['S_Value_part1', 'As1', 'S_Value_part2', 'As2', 'S_Value_res']]

    return result_df

def parse_tsi_value(filename='tsi_value.csv'):
    file = os.path.join("data", filename)
    df = pd.read_csv(file)
    df['TSI_Value_part1'] = df['TSI_Value1'] * (df['%_1'] / 100)
    df['TSI_Value_part2'] = df['TSI_Value2'] * (df['%_2'] / 100)

    result_df = df[['TSI_Value_part1', 'TSI_Value_part2', 'TSI_Value_res']]

    return result_df

def parse_p_value(filename='p_value.csv'):
    file = os.path.join("data", filename)
    df = pd.read_csv(file)
    df['P_Value_part1'] = df['P_Value1'] * (df['%_1'] / 100)
    df['P_Value_part2'] = df['P_Value2'] * (df['%_2'] / 100)

    result_df = df[['P_Value_part1', 'P_Value_part2', 'P_Value_res']]

    return result_df