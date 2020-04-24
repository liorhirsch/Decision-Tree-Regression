import numpy as np
import pandas as pd
from sklearn.linear_model._base import LinearModel

from src.DataTypesHandlers.CategoricalHandler import CategoricalHandler
from src.DataTypesHandlers.NumberHandler import NumberHandler
from src.TreeNodes.BaseTreeNode import BaseTreeNode
from src.TreeNodes.TreeNodeNumerical import TreeNodeNumerical
from src.utils import fit_predict_lr_model, get_data_without_categorical_cols


class DecisionTree():
    linear_regression_model: LinearModel
    min_elememnts_in_node: int

    def __init__(self, linear_regression_model, min_elememnts_in_node=5):
        self.linear_regression_model = linear_regression_model
        self.min_elememnts_in_node = min_elememnts_in_node
        pass

    def fit(self, X, Y):
        self.root = self.split(X, Y)

    def split(self, X, Y):
        original_lr, original_mse = fit_predict_lr_model(X, Y, self.linear_regression_model)
        best_tn: BaseTreeNode = self.try_all_splits_for_col(X, Y, original_mse)

        if best_tn == None:
            btn = BaseTreeNode(None, [], [original_lr], None)
            btn.self_lr_model = original_lr
            return btn

        best_tn.self_lr_model = original_lr
        dataset = pd.concat([X, Y], axis=1)
        target_col = dataset.columns[-1]
        for idx in range(len(best_tn.dataset_groups)):
            curr_dataset = best_tn.dataset_groups[idx]
            curr_left_x, curr_left_y = self.extract_x_y(curr_dataset, target_col)
            child_tn = self.split(curr_left_x, curr_left_y)
            best_tn.children_nodes.append(child_tn)

        return best_tn

    def extract_x_y(self, dataset, target_col):
        dataset_x = dataset.drop(columns=[target_col])
        dataset_y = dataset[target_col]
        return dataset_x, dataset_y

    def try_all_splits_for_col(self, X, Y, prev_mse) -> BaseTreeNode:
        dataset = pd.concat([X, Y], axis=1)
        best_mse_val = prev_mse
        best_tn = None

        number_handler = NumberHandler(dataset, self.linear_regression_model, self.min_elememnts_in_node)
        categorical_handler = CategoricalHandler(dataset, self.linear_regression_model, self.min_elememnts_in_node)

        for col in X.columns:
            tn = None
            if X[col].dtype == np.int64 or X[col].dtype == np.float64:
                chosen_linear_regression, chosen_mse_val, best_split, chosen_col_val = number_handler.split(col)
                if len(best_split) > 0:
                    tn = number_handler.build_tree_node(chosen_col_val, best_split, chosen_linear_regression, col)

            else:
                groups_to_lr_model, chosen_mse_val, group_to_dataset = categorical_handler.split(col)

                if group_to_dataset is not None and len(group_to_dataset) > 0:
                    tn = categorical_handler.build_tree_node(groups_to_lr_model, group_to_dataset, col)

            if tn is not None and best_mse_val > chosen_mse_val:
                best_tn = tn
                best_mse_val = chosen_mse_val

        return best_tn

    def predict(self, x):
        x_to_pred = get_data_without_categorical_cols(x)
        return [self.find_leaf_node(row, x_to_pred.loc[i], self.root) for i, row in x.iterrows()]

    def find_leaf_node(self, x, x_to_pred, node: BaseTreeNode):
        lr_model, child_node = node.get_match_lr_child(x)

        if child_node is None:
            if lr_model is None:
                return node.self_lr_model.predict([x_to_pred])
            else:
                return lr_model.predict([x_to_pred])
        else:
            val = self.find_leaf_node(x, x_to_pred, child_node)
            if val is None:
                return lr_model.predict([x_to_pred])
            else:
                return val
