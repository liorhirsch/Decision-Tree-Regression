from sklearn.metrics import mean_squared_error
import pandas as pd

def get_data_without_categorical_cols(X):
    return X.select_dtypes(exclude=['category'])

def sort_columns_names_and_get_dummies(df):
    df.columns = df.columns.map(str)
    df_sorted_columns = pd.get_dummies(df)
    return df_sorted_columns.reindex(sorted(df_sorted_columns.columns), axis=1)


def fit_predict_lr_model(x, y, linear_regression_model):
    model = linear_regression_model()

    # x_no_categorical = get_data_without_categorical_cols(x)
    x_to_learn = sort_columns_names_and_get_dummies(x)
    model.fit(x_to_learn, y)
    preds = model.predict(x_to_learn)
    mse = mean_squared_error(preds, y)
    return model, mse
