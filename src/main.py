import pandas as pd
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.DecisionTree import DecisionTree

data = pd.read_table("../Data/airfoil_self_noise.dat", header=None)
X = data.drop(columns=5)
Y = data[5]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

dt = DecisionTree(Ridge)
dt.fit(x_train, y_train)

preds = dt.predict(x_test)
mean_squared_error(y_test, preds)


