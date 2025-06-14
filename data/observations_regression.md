
# Cechy warte sprawdzenia:
Density
S

# TSI:

srednia z 50:
samo t value z augmentacją n_augmentations = 35, noise 0.05,  0.75 AdaBoostRegressor lub MLPRegressor
innym razem z augmentacją n_augmentations = 35, noise 0.05 i bez As, 0.82 z RadiusNeighborsRegressor, 0.8 z MLPRegressor

t value + Density z augmentacją n_augmentations = 35, noise 0.05,  0.7849 ExtraTreesRegressor, MLPRegressor dalej 0.75
t value + Density + As z augmentacją n_augmentations = 35, noise 0.05,  0.7849 ExtraTreesRegressor, MLPRegressor dalej 0.75, 0.9007 innym razem 0.8687 z RadiusNeighborsRegressor??
t value + As z augmentacją n_augmentations = 35, noise 0.05, 0.88 z RadiusNeighborsRegressor, 0.85 z ExtraTreesRegressor
wiekszy noise - 0.72

z CII wyniki takie same jak z samym t_val


### walidacja krzyzowa 5 fold 20 repeat 
featureset [['TSI_Value_part1', 'TSI_Value_part2', 'TSI_Value_res', 'D1_scaled', 'D2_scaled', 'As1_scaled', 'As2_scaled']]
1. AdaBoostRegressor: Mean R^2 = 0.9274 (±0.0771), Mean RMSE = 0.4285 (±0.1501)
2. GradientBoostingRegressor: Mean R^2 = 0.8738 (±0.1707), Mean RMSE = 0.5781 (±0.3773)
3. ExtraTreesRegressor: Mean R^2 = 0.8698 (±0.1446), Mean RMSE = 0.5721 (±0.2662)

featureset [['TSI_Value_part1', 'TSI_Value_part2', 'TSI_Value_res', 'D1_scaled', 'D2_scaled']]
0.7 best
najlepsze hiperparametry dla extratrees:
Best R^2: 0.7918
Best Params: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 50}
Best RMSE (mean ± std): 0.7379 ± 0.3152

# p-value:

srednia z 50:
bez augmentacji i z As jako cechą - 0.80
p-value z augmentacją n_augmentations = 35, noise 0.05 i bez As - 0.73 ExtraTreesRegressor
p-value + As z augmentacją n_augmentations = 35, noise 0.05 i z As jako cechą - 0.87 ExtraTreesRegressor

z usunięciem wwartości z p-value = 0 - 0.72

dodanie Density nic nie daje sensownego (dla braku zerowych p-value, inaczej nie było testowane)


### walidacja krzyzowa 5 fold 20 repeat

featureset [['P_Value_part1', 'P_Value_part2', 'P_Value_res']]
1. ExtraTreesRegressor: Mean R^2 = 0.8114 (±0.1373), Mean RMSE = 0.2099 (±0.0491)
2. GradientBoostingRegressor: Mean R^2 = 0.7684 (±0.1482), Mean RMSE = 0.2344 (±0.0627)
3. BaggingRegressor: Mean R^2 = 0.7410 (±0.1537), Mean RMSE = 0.2517 (±0.0716)

# SARA szukanie CII
uzywajac features = [
        'D1_scaled', 'As1_scaled',  'S1_scaled', 'R1_scaled', 'Ar1_scaled',
        'D2_scaled', 'As2_scaled', 'S2_scaled', 'R2_scaled', 'Ar2_scaled'
    ]
1. MLPRegressor: Mean R^2 = 0.9146, Mean RMSE = 0.1586
2. NuSVR: Mean R^2 = 0.9145, Mean RMSE = 0.1589
3. AdaBoostRegressor: Mean R^2 = 0.8706, Mean RMSE = 0.1951

uzywając dodatkowo typów próbek ropy:
1. NuSVR: Mean R^2 = 0.9203, Mean RMSE = 0.1294
0.89 z typami i tylko As i S

# s-value
TODO bo cos jest nie tak
!moze trzeba usunac outlierów
!hiperparametry
0.65 z As, bez tragedia

typy z As 0.5
typy bez As 0.6-0.7