
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

def parse_s_value(filename='s_value_As_types_s.csv'):
    file = os.path.join("data", filename)
    df = pd.read_csv(file)
    
    from utils.augmentation import augment_data
    columns_to_augment = ['S_Value1', 'S_Value2', 'As1', 'As2', '%_1', '%_2', 'S_Value_res', 'Type1', 'Type2', 'S1', 'S2']
    dfa = augment_data(df, columns_to_augment, 3, 35, 0.05)
    df = pd.concat([dfa, df])
    
    df['S_Value_part1'] = df['S_Value1'] * (df['%_1'] / 100)
    df['S_Value_part2'] = df['S_Value2'] * (df['%_2'] / 100)
    df['As1_scaled'] = df['As1'] * (df['%_1'] / 100)
    df['As2_scaled'] = df['As2'] * (df['%_2'] / 100)
    df['S1_scaled'] = df['S1'] * (df['%_1'] / 100)
    df['S2_scaled'] = df['S2'] * (df['%_2'] / 100)

    result_df = df[['S_Value_part1', 'S_Value_part2', 'S_Value_res', 'As1_scaled', 'As2_scaled']]

    return result_df

def parse_tsi_value(filename='tsi_value_with_As_Density.csv'):
    file = os.path.join("data", filename)
    df = pd.read_csv(file)
    
    from utils.augmentation import augment_data
    columns_to_augment = ['TSI_Value1', 'TSI_Value2', 'As1', 'As2', '%_1', '%_2', 'TSI_Value_res', 'D1', 'D2']
    dfa = augment_data(df, columns_to_augment, 3, 35, 0.05)
    df = pd.concat([dfa, df])
    
    df['TSI_Value_part1'] = df['TSI_Value1'] * (df['%_1'] / 100)
    df['TSI_Value_part2'] = df['TSI_Value2'] * (df['%_2'] / 100)
    df['As1_scaled'] = df['As1'] * (df['%_1'] / 100)
    df['As2_scaled'] = df['As2'] * (df['%_2'] / 100)
    df['D1_scaled'] = df['D1'] * (df['%_1'] / 100)
    df['D2_scaled'] = df['D2'] * (df['%_2'] / 100)
    
    result_df = df[['TSI_Value_part1', 'TSI_Value_part2', 'TSI_Value_res', 'D1_scaled', 'D2_scaled']]

    return result_df

def parse_p_value(filename='p_value_with_As_Density.csv'):
    file = os.path.join("data", filename)
    df = pd.read_csv(file)
    
    from utils.augmentation import augment_data
    columns_to_augment = ['P_Value1', 'P_Value2', 'As1', 'As2', '%_1', '%_2', 'P_Value_res', 'D1', 'D2']
    
    df['P_Value_part1'] = df['P_Value1'] * (df['%_1'] / 100)
    df['P_Value_part2'] = df['P_Value2'] * (df['%_2'] / 100)
    df['As1_scaled'] = df['As1'] * (df['%_1'] / 100)
    df['As2_scaled'] = df['As2'] * (df['%_2'] / 100)
    df['D1_scaled'] = df['D1'] * (df['%_1'] / 100)
    df['D2_scaled'] = df['D2'] * (df['%_2'] / 100)
    
    result_df = df[['P_Value_part1', 'P_Value_part2', 'P_Value_res']]

    return result_df

def logit(p, eps=1e-6):
    import numpy as np
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def parse_asmix(filename='mieszaniny.csv'):
    file = os.path.join("data", filename)
    df = pd.read_csv(file)
    
    from utils.augmentation import augment_data
    columns_to_augment = [
        'S1', 'Ar1', 'R1', 'As1', '%1',
        'S2', 'Ar2', 'R2', 'As2', '%2', 'AsMix'
    ]
    dfa = augment_data(df, columns_to_augment, 3, 15, 0.05)
    df = pd.concat([dfa, df])
    
    df['S1_scaled'] = df['S1'] * (df['%1'] / 100)
    df['Ar1_scaled'] = df['Ar1'] * (df['%1'] / 100)
    df['R1_scaled'] = df['R1'] * (df['%1'] / 100)
    df['As1_scaled'] = df['As1'] * (df['%1'] / 100)
    df['S2_scaled'] = df['S2'] * (df['%2'] / 100)
    df['Ar2_scaled'] = df['Ar2'] * (df['%2'] / 100)
    df['R2_scaled'] = df['R2'] * (df['%2'] / 100)
    df['As2_scaled'] = df['As2'] * (df['%2'] / 100)
    
    #todo typ ropy
    features = [
        'As1_scaled', '%1', 'S1_scaled', 'R1_scaled', 'Ar1_scaled',
        'As2_scaled', '%2', 'S2_scaled', 'R2_scaled', 'Ar2_scaled'
    ]
    target = 'AsMix'
    result_df = df[features + [target]]
    return result_df


def parse_asmix_with_density(filename='mieszaniny_sara_with_density.csv'):
    file = os.path.join("data", filename)
    df = pd.read_csv(file)
    target = 'AsMix'
    
    from utils.augmentation import augment_data
    columns_to_augment = [
        'S1', 'Ar1', 'R1', 'As1', '%1', 'D1', 'D2',
        'S2', 'Ar2', 'R2', 'As2', '%2', 'AsMix'
    ]
    dfa = augment_data(df, columns_to_augment, 3, 25, 0.05)
    df = pd.concat([dfa, df])
    
    df['D1_scaled'] = df['D1'] * (df['%1'] / 100)
    df['D2_scaled'] = df['D2'] * (df['%2'] / 100)
    
    df['S1_scaled'] = df['S1'] * (df['%1'] / 100)
    df['Ar1_scaled'] = df['Ar1'] * (df['%1'] / 100)
    df['R1_scaled'] = df['R1'] * (df['%1'] / 100)
    df['As1_scaled'] = df['As1'] * (df['%1'] / 100)
    df['S2_scaled'] = df['S2'] * (df['%2'] / 100)
    df['Ar2_scaled'] = df['Ar2'] * (df['%2'] / 100)
    df['R2_scaled'] = df['R2'] * (df['%2'] / 100)
    df['As2_scaled'] = df['As2'] * (df['%2'] / 100)
    
    #todo typ ropy
    features = [
        'D1_scaled', 'As1_scaled',  'S1_scaled', 'R1_scaled', 'Ar1_scaled',
        'D2_scaled', 'As2_scaled', 'S2_scaled', 'R2_scaled', 'Ar2_scaled'
    ]
    result_df = df[features + [target]]
    return result_df

def parse_asmix_with_density_find_CII(filename='mieszaniny_sara_types_appended.csv'):
    file = os.path.join("data", filename)
    df = pd.read_csv(file)
    target = 'CII'
    
    from utils.augmentation import augment_data
    columns_to_augment = [
        'S1', 'Ar1', 'R1', 'As1', '%1', 'D1', 'D2', 'Type1',
        'S2', 'Ar2', 'R2', 'As2', '%2', target, 'Type2'
    ]
    dfa = augment_data(df, columns_to_augment, 3, 15, 0.05)
    df = pd.concat([dfa, df])
    
    df['D1_scaled'] = df['D1'] * (df['%1'] / 100)
    df['D2_scaled'] = df['D2'] * (df['%2'] / 100)
    
    df['S1_scaled'] = df['S1'] * (df['%1'] / 100)
    df['Ar1_scaled'] = df['Ar1'] * (df['%1'] / 100)
    df['R1_scaled'] = df['R1'] * (df['%1'] / 100)
    df['As1_scaled'] = df['As1'] * (df['%1'] / 100)
    df['S2_scaled'] = df['S2'] * (df['%2'] / 100)
    df['Ar2_scaled'] = df['Ar2'] * (df['%2'] / 100)
    df['R2_scaled'] = df['R2'] * (df['%2'] / 100)
    df['As2_scaled'] = df['As2'] * (df['%2'] / 100)
    
    #todo typ ropy
    features = [
        'D1_scaled', 'As1_scaled',  'S1_scaled', 'R1_scaled', 'Ar1_scaled', 'Type1',
        'D2_scaled', 'As2_scaled', 'S2_scaled', 'R2_scaled', 'Ar2_scaled', 'Type2'
    ]
    result_df = df[features + [target]]
    return result_df

def generate_types():
    import joblib

    clf = joblib.load('models\\random forest_density_group.pkl')

    df = pd.read_csv('data\\mieszaniny_sara_with_density copy.csv')

    features = ['D', 'S', 'Ar', 'R', 'As']

    oil1 = df[['D1', 'S1', 'Ar1', 'R1', 'As1']].copy()
    oil1.columns = ['Density', 'S', 'Ar', 'R', 'As']

    oil2 = df[['D2', 'S2', 'Ar2', 'R2', 'As2']].copy()
    oil2.columns = ['Density', 'S', 'Ar', 'R', 'As']

    df['Type1'] = clf.predict(oil1)
    df['Type2'] = clf.predict(oil2)

    df.to_csv('classified_output.csv', index=False)