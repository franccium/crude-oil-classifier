#def sara_linear_regression_train():
#     df = parse_sara('sara.csv')
#     X = df.drop(columns=['CII', 'ID_1', 'ID_2', '%_1', '%_2'])
#     y = df['CII']
#
#     regr = LinearRegression()
#     regr.fit(X, y)
#
#     joblib.dump(regr, 'sara_linear_regression.pkl')
#
#
#
#
# sara_linear_regression_train()