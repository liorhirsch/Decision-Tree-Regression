import pandas as pd
from sklearn.linear_model import SGDRegressor, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

from src.DecisionTree import DecisionTree
from src.TreeNodes.BaseTreeNode import BaseTreeNode
from src.utils import fit_predict_lr_model


def load_airfoil_data():
    return pd.read_table("../Data/airfoil_self_noise.dat", header=None)

def load_servo_data():
    df = pd.read_csv("../Data/servo.data", header=None)
    for col in df.columns:
        if df[col].dtype == np.object:
            df[col] = df[col].astype('category')
    return df

def load_3d_spatial():
    df = pd.read_csv("../Data/Container_Crane_Controller_Data_Set.csv")
    return df

def load_qsar_aquatic_toxicity():
    df = pd.read_csv("../Data/qsar_aquatic_toxicity.csv", header=None)
    return df

def load_Concrete_Data():
    df = pd.read_excel("../Data/Concrete_Data.xls")
    return df

def load_machine():
    df = pd.read_csv("../Data/machine.data", header=None)
    for col in df.columns:
        if df[col].dtype == np.object:
            df[col] = df[col].astype('category')
    return df


def check_tree_height(node: BaseTreeNode):
    if len(node.children_nodes) == 0:
        return 1

    max_tree_height = 0
    for child in node.children_nodes:
        max_tree_height = max(check_tree_height(child), max_tree_height)

    return max_tree_height + 1

#
# datas = [load_servo_data(), load_airfoil_data(), load_3d_spatial(), load_Concrete_Data(), load_qsar_aquatic_toxicity()]
# datas = [load_machine()]
# leaf_size = range(5,16,5)
# lr_models = [Ridge, LinearRegression, SGDRegressor]
# tests = []
# for idx, data in enumerate(datas):
#     curr_dataset_test = []
#     print("===========================================================")
#     target_col = data.columns[-1]
#     X = data.drop(columns=target_col)
#     X.columns = X.columns.map(str)
#     Y = data[target_col]
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
#
#     for curr_leaf_size in leaf_size:
#         for curr_lr_model in lr_models:
#             dt = DecisionTree(curr_lr_model, curr_leaf_size)
#             dt.fit(x_train, y_train)
#
#             if (dt.root is not None):
#                 max_height = check_tree_height(dt.root)
#                 preds = dt.predict(x_test)
#                 # print("Dataset {}, leaf num {}, lr_model {} ".format(str(idx), str(curr_leaf_size), str(curr_lr_model)))
#                 mse_tree = mean_squared_error(y_test, preds)
#                 # print("With Decision Tree: ", mse_tree)
#                 _, original_mse = fit_predict_lr_model(X, Y, curr_lr_model)
#
#                 # print("Without tree: ", original_mse)
#                 # print("-------------------------------------------------------")
#                 curr_dataset_test.append((curr_leaf_size, curr_lr_model, mse_tree, original_mse))
#             else:
#                 print("No Tree")
#
#     tests.append(curr_dataset_test)
#
#
#
#
