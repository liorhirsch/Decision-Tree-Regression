from sklearn.metrics import mean_squared_error

from src.TreeNodes.BaseTreeNode import BaseTreeNode
from src.utils import fit_predict_lr_model


class BaseHandler():
    def __init__(self, dataset, linear_regression_model, min_elements_in_node):
        self.dataset = dataset
        self.linear_regression_model = linear_regression_model
        self.min_elememnts_in_node = min_elements_in_node


    def split(self, col):
        pass

    def fit_lr_model(self, dataset):
        target_col = dataset.columns[-1]
        x = dataset.drop(columns=[target_col])
        y = dataset[target_col]
        lr_model, mse = fit_predict_lr_model(x, y, self.linear_regression_model)
        return lr_model, mse

    def build_tree_node(self) -> BaseTreeNode:
        pass

