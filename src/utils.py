from sklearn.metrics import mean_squared_error


def get_data_without_categorical_cols(X):
    return X.select_dtypes(exclude=['category'])


def fit_predict_lr_model(x, y, linear_regression_model):
    model = linear_regression_model()
    x_no_categorical = get_data_without_categorical_cols(x)
    model.fit(x_no_categorical, y)
    preds = model.predict(x_no_categorical)
    mse = mean_squared_error(preds, y)
    return model, mse