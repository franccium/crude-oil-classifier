
import os
import re
import pandas as pd

def parse_s_value(filename='s_value_As_types_s.csv'):
    file = os.path.join("data", filename)
    df = pd.read_csv(file)
    
    from utils.augmentation import augment_data
    columns_to_augment = ['S_Value1', 'S_Value2', 'As1', 'As2', '%_1', '%_2', 'S_Value_res', 'Type1', 'Type2', 'S1', 'S2']
    dfa = augment_data(df, columns_to_augment, 3, 35, 0.05)
    df = pd.concat([dfa, df])
    
    df['S_Value_part1'] = df['S_Value1'] * (df['%_1'])
    df['S_Value_part2'] = df['S_Value2'] * (df['%_2'])
    df['As1_scaled'] = df['As1'] * (df['%_1'])
    df['As2_scaled'] = df['As2'] * (df['%_2'])
    df['S1_scaled'] = df['S1'] * (df['%_1'])
    df['S2_scaled'] = df['S2'] * (df['%_2'])
    
    df['S_Value_sum'] = df['S_Value_part1'] + df['S_Value_part2']
    df['S_Value_mean'] = (df['S_Value_part1'] + df['S_Value_part2']) / 2
    df['S_Value_min'] = df[['S_Value_part1', 'S_Value_part2']].min(axis=1)
    df['S_Value_max'] = df[['S_Value_part1', 'S_Value_part2']].max(axis=1)

    df['As_scaled_sum'] = df['As1_scaled'] + df['As2_scaled']
    df['As_scaled_mean'] = (df['As1_scaled'] + df['As2_scaled']) / 2
    df['As_scaled_min'] = df[['As1_scaled', 'As2_scaled']].min(axis=1)
    df['As_scaled_max'] = df[['As1_scaled', 'As2_scaled']].max(axis=1)

    #result_df = df[['S_Value_part1', 'S_Value_part2', 'S_Value_res', 'As1_scaled', 'As2_scaled']]
    result_df = df[['S_Value_mean', 'As_scaled_mean', 'S_Value_sum', 'S_Value_res']]
    
    return result_df

def parse_tsi_value(filename='tsi_value_with_As_Density.csv'):
    file = os.path.join("data", filename)
    df = pd.read_csv(file)
    
    from utils.augmentation import augment_data
    columns_to_augment = ['TSI_Value1', 'TSI_Value2', 'As1', 'As2', '%_1', '%_2', 'TSI_Value_res', 'D1', 'D2']
    dfa = augment_data(df, columns_to_augment, 3, 35, 0.05)
    df = pd.concat([dfa, df])
    
    df['TSI_Value_part1'] = df['TSI_Value1'] * (df['%_1'])
    df['TSI_Value_part2'] = df['TSI_Value2'] * (df['%_2'])
    df['As1_scaled'] = df['As1'] * (df['%_1'])
    df['As2_scaled'] = df['As2'] * (df['%_2'])
    df['D1_scaled'] = df['D1'] * (df['%_1'])
    df['D2_scaled'] = df['D2'] * (df['%_2'])

    df['TSI_Value_sum'] = df['TSI_Value_part1'] + df['TSI_Value_part2']
    df['TSI_Value_mean'] = (df['TSI_Value_part1'] + df['TSI_Value_part2']) / 2
    df['TSI_Value_min'] = df[['TSI_Value_part1', 'TSI_Value_part2']].min(axis=1)
    df['TSI_Value_max'] = df[['TSI_Value_part1', 'TSI_Value_part2']].max(axis=1)

    df['As_scaled_sum'] = df['As1_scaled'] + df['As2_scaled']
    df['As_scaled_mean'] = (df['As1_scaled'] + df['As2_scaled']) / 2
    df['As_scaled_min'] = df[['As1_scaled', 'As2_scaled']].min(axis=1)
    df['As_scaled_max'] = df[['As1_scaled', 'As2_scaled']].max(axis=1)

    df['D_scaled_sum'] = df['D1_scaled'] + df['D2_scaled']
    df['D_scaled_mean'] = (df['D1_scaled'] + df['D2_scaled']) / 2
    df['D_scaled_min'] = df[['D1_scaled', 'D2_scaled']].min(axis=1)
    df['D_scaled_max'] = df[['D1_scaled', 'D2_scaled']].max(axis=1)
    
    #result_df = df[['TSI_Value_part1', 'TSI_Value_part2', 'TSI_Value_res', 'As1_scaled', 'As2_scaled']]

    result_df = df[['TSI_Value_mean', 'As_scaled_mean', 'D_scaled_mean','TSI_Value_res', 'TSI_Value_sum']]
    return result_df

def parse_p_value(filename='p_value_with_As_Density.csv'):
    file = os.path.join("data", filename)
    df = pd.read_csv(file)
    
    from utils.augmentation import augment_data
    columns_to_augment = ['P_Value1', 'P_Value2', 'As1', 'As2', '%_1', '%_2', 'P_Value_res', 'D1', 'D2']
    dfa = augment_data(df, columns_to_augment, 3, 35, 0.03)
    df = pd.concat([dfa, df])
    
    df['P_Value_part1'] = df['P_Value1'] * (df['%_1'])
    df['P_Value_part2'] = df['P_Value2'] * (df['%_2'])
    df['As1_scaled'] = df['As1'] * (df['%_1'])
    df['As2_scaled'] = df['As2'] * (df['%_2'])
    df['D1_scaled'] = df['D1'] * (df['%_1'])
    df['D2_scaled'] = df['D2'] * (df['%_2'])
    
    #result_df = df[['P_Value_part1', 'P_Value_part2', 'P_Value_res', 'As1_scaled', 'As2_scaled', 'D1_scaled', 'D2_scaled']]
    
    df['P_Value_sum'] = df['P_Value_part1'] + df['P_Value_part2']
    df['P_Value_mean'] = (df['P_Value_part1'] + df['P_Value_part2']) / 2
    df['P_Value_min'] = df[['P_Value_part1', 'P_Value_part2']].min(axis=1)
    df['P_Value_max'] = df[['P_Value_part1', 'P_Value_part2']].max(axis=1)

    df['As_scaled_sum'] = df['As1_scaled'] + df['As2_scaled']
    df['As_scaled_mean'] = (df['As1_scaled'] + df['As2_scaled']) / 2
    df['As_scaled_min'] = df[['As1_scaled', 'As2_scaled']].min(axis=1)
    df['As_scaled_max'] = df[['As1_scaled', 'As2_scaled']].max(axis=1)

    df['D_scaled_sum'] = df['D1_scaled'] + df['D2_scaled']
    df['D_scaled_mean'] = (df['D1_scaled'] + df['D2_scaled']) / 2
    df['D_scaled_min'] = df[['D1_scaled', 'D2_scaled']].min(axis=1)
    df['D_scaled_max'] = df[['D1_scaled', 'D2_scaled']].max(axis=1)

    result_df = df[['P_Value_mean', 'P_Value_sum','As_scaled_sum', 'As_scaled_mean','D_scaled_sum', 'D_scaled_mean', 'P_Value_res']]

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
    
    df['D1_scaled'] = df['D1'] * (df['%1'])
    df['D2_scaled'] = df['D2'] * (df['%2'])
    
    df['S1_scaled'] = df['S1'] * (df['%1'])
    df['Ar1_scaled'] = df['Ar1'] * (df['%1'])
    df['R1_scaled'] = df['R1'] * (df['%1'])
    df['As1_scaled'] = df['As1'] * (df['%1'])
    df['S2_scaled'] = df['S2'] * (df['%2'])
    df['Ar2_scaled'] = df['Ar2'] * (df['%2'])
    df['R2_scaled'] = df['R2'] * (df['%2'])
    df['As2_scaled'] = df['As2'] * (df['%2'])
    
    features = [
        'D1_scaled', 'As1_scaled',  'S1_scaled', 'R1_scaled', 'Ar1_scaled',
        'D2_scaled', 'As2_scaled', 'S2_scaled', 'R2_scaled', 'Ar2_scaled',
    ]
    result_df = df[features + [target]]
    return result_df