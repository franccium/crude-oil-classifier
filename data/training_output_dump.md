


# SUMMARY

## NO AUGMENTATION, DENSITY AND CII
Model              CV=5    CV=6    CV=7
MLP               0.960   0.979   0.939
KNN               0.822   0.866   0.867
SVC               0.900   0.926   0.944
Decision Tree     0.764   0.826   0.847
Random Forest     0.920   0.924   0.923

## WITH AUGMENTATION, DENSITY AND CII
Model              CV=5    CV=6    CV=7
MLP               0.779   0.791   0.777
KNN               0.792   0.815   0.827
SVC               0.815   0.815   0.814
Decision Tree     0.840   0.829   0.828
Random Forest     0.815   0.790   0.801

## NO AUGMENTATION, DENSITY AND S Ar R As
Model              CV=5    CV=6    CV=7
MLP               0.884   0.863   0.880
KNN               0.687   0.667   0.730
SVC               0.884   0.903   0.923
Decision Tree     0.884   0.905   0.901
Random Forest     0.920   0.942   0.941

## WITH AUGMENTATION, DENSITY AND S Ar R As
Model              CV=5    CV=6    CV=7
MLP               0.840   0.815   0.814
KNN               0.842   0.853   0.828
SVC               0.765   0.764   0.775
Decision Tree     0.766   0.753   0.752
Random Forest     0.828   0.815   0.827

## NO AUGMENTATION, DENSITY, S Ar R As AND CII
Model              CV=5    CV=6    CV=7
MLP               0.884   0.921   0.921
KNN               0.607   0.609   0.668
SVC               0.884   0.884   0.923
Decision Tree     0.920   0.942   0.944
Random Forest     0.940   0.900   0.941

## WITH AUGMENTATION, DENSITY, S Ar R As AND CII
Model              CV=5    CV=6    CV=7
MLP               0.853   0.841   0.864
KNN               0.817   0.828   0.815
SVC               0.779   0.765   0.776
Decision Tree     0.779   0.779   0.742
Random Forest     0.840   0.828   0.828



## NO AUGMENTATION, DENSITY AND CII
============================================================
Mean CV accuracy vs split count for each model
============================================================

Model              CV=5    CV=6    CV=7
MLP               0.960   0.979   0.939
KNN               0.822   0.866   0.867
SVC               0.900   0.926   0.944
Decision Tree     0.764   0.826   0.847
Random Forest     0.920   0.924   0.923
============================================================

============================================================
Per-class mean precision for each model and split count
============================================================

Model                 light      medium       heavy
MLP             (CV=5)       1.000       0.900       0.950
MLP             (CV=6)       1.000       0.900       1.000
MLP             (CV=7)       0.952       0.800       1.000
KNN             (CV=5)       0.810       1.000       0.750
KNN             (CV=6)       0.810       1.000       0.850
KNN             (CV=7)       0.810       1.000       0.850
SVC             (CV=5)       0.952       0.800       0.900
SVC             (CV=6)       0.952       0.900       0.900
SVC             (CV=7)       1.000       0.900       0.900
Decision Tree   (CV=5)       0.952       0.400       0.750
Decision Tree   (CV=6)       0.952       0.600       0.800
Decision Tree   (CV=7)       0.952       0.700       0.800
Random Forest   (CV=5)       0.952       0.900       0.900
Random Forest   (CV=6)       0.952       0.900       0.900
Random Forest   (CV=7)       0.952       0.900       0.900
============================================================

## WITH AUGMENTATION, DENSITY AND CII

============================================================
Mean CV accuracy vs split count for each model
============================================================

Model              CV=5    CV=6    CV=7
MLP               0.779   0.791   0.777
KNN               0.792   0.815   0.827
SVC               0.815   0.815   0.814
Decision Tree     0.840   0.829   0.828
Random Forest     0.815   0.790   0.801
============================================================

============================================================
Per-class mean precision for each model and split count
============================================================

Model                 light      medium       heavy
MLP             (CV=5)       0.806       0.800       0.733
MLP             (CV=6)       0.806       0.800       0.767
MLP             (CV=7)       0.839       0.750       0.733
KNN             (CV=5)       0.742       0.850       0.800
KNN             (CV=6)       0.774       0.850       0.833
KNN             (CV=7)       0.806       0.850       0.833
SVC             (CV=5)       0.871       0.800       0.767
SVC             (CV=6)       0.871       0.800       0.767
SVC             (CV=7)       0.871       0.800       0.767
Decision Tree   (CV=5)       0.806       0.750       0.933
Decision Tree   (CV=6)       0.806       0.750       0.900
Decision Tree   (CV=7)       0.806       0.750       0.900
Random Forest   (CV=5)       0.839       0.750       0.833
Random Forest   (CV=6)       0.839       0.750       0.767
Random Forest   (CV=7)       0.839       0.750       0.800
============================================================

## NO AUGMENTATION, DENSITY AND S Ar R As

============================================================
Mean CV accuracy vs split count for each model
============================================================

Model              CV=5    CV=6    CV=7
MLP               0.884   0.863   0.880
KNN               0.687   0.667   0.730
SVC               0.884   0.903   0.923
Decision Tree     0.884   0.905   0.901
Random Forest     0.920   0.942   0.941
============================================================

============================================================
Per-class mean precision for each model and split count
============================================================

Model                 light      medium       heavy
MLP             (CV=5)       0.905       1.000       0.800
MLP             (CV=6)       0.905       1.000       0.750
MLP             (CV=7)       0.952       1.000       0.750
KNN             (CV=5)       0.619       1.000       0.600
KNN             (CV=6)       0.571       1.000       0.600
KNN             (CV=7)       0.619       1.000       0.700
SVC             (CV=5)       0.905       1.000       0.800
SVC             (CV=6)       0.857       1.000       0.900
SVC             (CV=7)       0.905       1.000       0.900
Decision Tree   (CV=5)       0.952       0.700       0.900
Decision Tree   (CV=6)       0.952       0.900       0.850
Decision Tree   (CV=7)       0.952       0.900       0.850
Random Forest   (CV=5)       0.952       0.900       0.900
Random Forest   (CV=6)       0.952       0.900       0.950
Random Forest   (CV=7)       0.952       0.900       0.950
============================================================

## WITH AUGMENTATION, DENSITY AND S Ar R As

============================================================
Mean CV accuracy vs split count for each model
============================================================

Model              CV=5    CV=6    CV=7
MLP               0.840   0.815   0.814
KNN               0.842   0.853   0.828
SVC               0.765   0.764   0.775
Decision Tree     0.766   0.753   0.752
Random Forest     0.828   0.815   0.827
============================================================

============================================================
Per-class mean precision for each model and split count
============================================================

Model                 light      medium       heavy
MLP             (CV=5)       0.871       0.850       0.800
MLP             (CV=6)       0.903       0.750       0.767
MLP             (CV=7)       0.935       0.800       0.700
KNN             (CV=5)       0.774       1.000       0.800
KNN             (CV=6)       0.839       1.000       0.767
KNN             (CV=7)       0.774       1.000       0.767
SVC             (CV=5)       0.806       0.850       0.667
SVC             (CV=6)       0.806       0.850       0.667
SVC             (CV=7)       0.806       0.850       0.700
Decision Tree   (CV=5)       0.806       0.850       0.667
Decision Tree   (CV=6)       0.774       0.850       0.667
Decision Tree   (CV=7)       0.774       0.850       0.667
Random Forest   (CV=5)       0.839       0.900       0.767
Random Forest   (CV=6)       0.839       0.850       0.767
Random Forest   (CV=7)       0.839       0.850       0.800
============================================================

## NO AUGMENTATION, DENSITY, S Ar R As AND CII

============================================================
Mean CV accuracy vs split count for each model
============================================================

Model              CV=5    CV=6    CV=7
MLP               0.884   0.921   0.921
KNN               0.607   0.609   0.668
SVC               0.884   0.884   0.923
Decision Tree     0.920   0.942   0.944
Random Forest     0.940   0.900   0.941
============================================================

============================================================
Per-class mean precision for each model and split count
============================================================

Model                 light      medium       heavy
MLP             (CV=5)       0.905       1.000       0.800
MLP             (CV=6)       0.952       1.000       0.850
MLP             (CV=7)       0.952       1.000       0.850
KNN             (CV=5)       0.619       0.800       0.500
KNN             (CV=6)       0.571       0.800       0.550
KNN             (CV=7)       0.571       0.800       0.700
SVC             (CV=5)       0.905       1.000       0.800
SVC             (CV=6)       0.810       1.000       0.900
SVC             (CV=7)       0.905       1.000       0.900
Decision Tree   (CV=5)       0.952       0.900       0.900
Decision Tree   (CV=6)       0.952       0.900       0.950
Decision Tree   (CV=7)       0.952       0.900       0.950
Random Forest   (CV=5)       0.952       0.900       0.950
Random Forest   (CV=6)       0.905       0.900       0.900
Random Forest   (CV=7)       0.952       0.900       0.950
============================================================

## WITH AUGMENTATION, DENSITY, S Ar R As AND CII

============================================================
Mean CV accuracy vs split count for each model
============================================================

Model              CV=5    CV=6    CV=7
MLP               0.853   0.841   0.864
KNN               0.817   0.828   0.815
SVC               0.779   0.765   0.776
Decision Tree     0.779   0.779   0.742
Random Forest     0.840   0.828   0.828
============================================================

============================================================
Per-class mean precision for each model and split count
============================================================

Model                 light      medium       heavy
MLP             (CV=5)       0.935       0.800       0.800
MLP             (CV=6)       0.935       0.800       0.767
MLP             (CV=7)       0.968       0.850       0.767
KNN             (CV=5)       0.742       1.000       0.767
KNN             (CV=6)       0.839       1.000       0.700
KNN             (CV=7)       0.774       1.000       0.733
SVC             (CV=5)       0.839       0.800       0.700
SVC             (CV=6)       0.839       0.800       0.667
SVC             (CV=7)       0.839       0.800       0.700
Decision Tree   (CV=5)       0.742       0.700       0.867
Decision Tree   (CV=6)       0.710       0.750       0.867
Decision Tree   (CV=7)       0.710       0.600       0.867
Random Forest   (CV=5)       0.871       0.900       0.767
Random Forest   (CV=6)       0.839       0.850       0.800
Random Forest   (CV=7)       0.839       0.900       0.767
============================================================


