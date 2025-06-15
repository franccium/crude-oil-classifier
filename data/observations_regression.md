
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
60 repeat:
1. AdaBoostRegressor: Mean R^2 = 0.8570 (±0.2408), Mean RMSE = 0.5595 (±0.2611)
2. ExtraTreesRegressor: Mean R^2 = 0.8462 (±0.1843), Mean RMSE = 0.6525 (±0.3157)


featureset [['TSI_Value_part1', 'TSI_Value_part2', 'TSI_Value_res', 'D1_scaled', 'D2_scaled']]
0.7 best
najlepsze hiperparametry dla extratrees:
Best R^2: 0.7918
Best Params: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 50}
Best RMSE (mean ± std): 0.7379 ± 0.3152


Tuning AdaBoostRegressor...
Best R^2: 0.7699
Best Params: {'learning_rate': 1.0, 'loss': 'linear', 'n_estimators': 100}
Best RMSE (mean ± std): 0.6569 ± 0.4164

Tuning ExtraTreesRegressor...
Best R^2: 0.7745
Best Params: {'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 50}
Best RMSE (mean ± std): 0.7680 ± 0.3429


# p-value:

srednia z 50:
bez augmentacji i z As jako cechą - 0.80
p-value z augmentacją n_augmentations = 35, noise 0.05 i bez As - 0.73 ExtraTreesRegressor
p-value + As z augmentacją n_augmentations = 35, noise 0.05 i z As jako cechą - 0.87 ExtraTreesRegressor

z usunięciem wwartości z p-value = 0 - 0.72

dodanie Density nic nie daje sensownego (dla braku zerowych p-value)


same p-value augmentacja 0.03 noise: 1. ExtraTreesRegressor: Mean R^2 = 0.7832 (±0.2531), Mean RMSE = 0.2129 (±0.0683)
moze byc dobre: Best Params: {'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100} --> teoretycznie 0.8 prtawie osiąga

Tuning ExtraTreesRegressor...
Best R^2: 0.7481
Best Params: {'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
Best RMSE (mean ± std): 0.2376 ± 0.0649


order invariant:
'P_Value_mean',
'As_scaled_sum', 'As_scaled_mean',
'D_scaled_sum', 'D_scaled_mean',
'P_Value_res'
1. ExtraTreesRegressor: Mean R^2 = 0.7639 (±0.1846), Mean RMSE = 0.2265 (±0.0512)
2. GradientBoostingRegressor: Mean R^2 = 0.7632 (±0.1824), Mean RMSE = 0.2277 (±0.0516)
3. BaggingRegressor: Mean R^2 = 0.7585 (±0.1795), Mean RMSE = 0.2311 (±0.0464)
4. RandomForestRegressor: Mean R^2 = 0.7584 (±0.1798), Mean RMSE = 0.2311 (±0.0459)
5. AdaBoostRegressor: Mean R^2 = 0.7410 (±0.1772), Mean RMSE = 0.2416 (±0.0518)


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
2. NuSVR: Mean R^2 = 0.9145, Mean RMSE = 0.1589 **najbardziej stabilny**, co ciekawe, skalowanie danych pogorszyło wyniki
3. AdaBoostRegressor: Mean R^2 = 0.8706, Mean RMSE = 0.1951

uzywając dodatkowo typów próbek ropy:
1. NuSVR: Mean R^2 = 0.9203, Mean RMSE = 0.1294
0.89 z typami i tylko As i S

# s-value
!moze trzeba usunac outlierów
!hiperparametry
0.65 z As, bez slabo

typy z As 0.5
typy bez As 0.6-0.7


Best R^2: 0.5672 Best Params: {'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
Best RMSE (mean ± std): 0.3523 ± 0.0800

Tuning ExtraTreesRegressor...
Best R^2: 0.4971
Best Params: {'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
Best RMSE (mean ± std): 0.3898 ± 0.0910


OUR REGRESSORS:
ensembles seemed to work best cause its all very random
done with RepeatedKFold(n_folds = 5, n_repeats = ensemble_size)
hyperparameters found via gridsearchCV

TSI - extra trees ensemble of 10
model = ExtraTreesRegressor(
            max_depth=8,
            min_samples_leaf=1,
            min_samples_split=5,
            n_estimators=200
        )

P-Val - extra trees ensemble of 10
 model = ExtraTreesRegressor(
            max_depth=8,
            min_samples_leaf=1,
            min_samples_split=2,
            n_estimators=100
        )

S-Val - extra trees ensemble of 1
 model = ExtraTreesRegressor(
            max_depth=8,
            min_samples_leaf=1,
            min_samples_split=2,
            n_estimators=100
        )

CII - NuSVR ensemble of 5
NuSVR(C=1, kernel='rbf', nu=0.5)
