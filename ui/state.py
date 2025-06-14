selected_file = "data_original.csv"
graph_flags = {
    'feature_space': True,
    'data_scatter': True,
    'cv_summary': True
}
mix = None
root = None
featureset = None
featureset_density_CII = ['Density', 'CII']
featureset_density_group = ['Density', 'S', 'Ar', 'R', 'As']
featureset_density_group_CII = ['Density', 'S', 'Ar', 'R', 'As', 'CII']
featureset_map = {0: featureset_density_CII, 1: featureset_density_group, 2: featureset_density_group_CII}
reverse_featureset_map = {tuple(v): k for k, v in featureset_map.items()}
best_model = None