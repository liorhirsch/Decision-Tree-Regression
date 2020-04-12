import numpy as np
import pandas as pd
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import mean_squared_error

from src.TreeNode import TreeNode


class DecisionTree():
    linear_regression_model: LinearModel
    min_elememnts_in_node: int

    def __init__(self, linear_regression_model, min_elememnts_in_node = 5):
        self.linear_regression_model = linear_regression_model
        self.min_elememnts_in_node = min_elememnts_in_node
        pass

    def fit(self, X, Y):
        self.root = self.split(X,Y)

    def split(self, X, Y):
        original_lr = self.linear_regression_model()
        original_lr.fit(X, Y)
        preds = original_lr.predict(X)
        original_mse = mean_squared_error(preds, Y)
        best_split, best_col, best_col_val, best_linear_regressions = self.try_all_splits_for_col(X, Y, original_mse)



        if len(best_split) == 0:
            return

        tn = TreeNode(best_col, best_col_val, best_linear_regressions[0], best_linear_regressions[1])

        left = best_split[0]
        left_x = left.drop(columns=[5])
        left_y = left[5]
        right = best_split[1]
        right_x = right.drop(columns=[5])
        right_y = right[5]

        left_tn = self.split(left_x, left_y)
        right_tn = self.split(right_x, right_y)

        tn.left = left_tn
        tn.right = right_tn

        return tn

    def try_all_splits_for_col(self, X, Y, prev_mse):
        # TODO merge X,Y
        dataset = pd.concat([X,Y], axis = 1)
        best_mse_val = prev_mse
        best_split = ()
        best_col = -1
        best_col_val = 0
        best_linear_regressions = ()

        for col in X.columns:
            for index, row in dataset.iterrows():
                split_value = row[col]

                left, right = self.split_dataset_for_cal_val(col, split_value, dataset)

                if (len(left) < self.min_elememnts_in_node or
                    len(right) < self.min_elememnts_in_node):
                    continue

                left_lr = self.linear_regression_model()
                left_lr.fit(left.drop(columns = [5]), left[5])
                preds = left_lr.predict(left.drop(columns = [5]))
                left_mse = mean_squared_error(preds, left[5])

                right_lr = self.linear_regression_model()
                right_lr.fit(right.drop(columns=[5]), right[5])
                preds = right_lr.predict(right.drop(columns=[5]))
                right_mse = mean_squared_error(preds, right[5])

                weighted_mse = (len(right) * right_mse + len(left) * left_mse) / len (dataset)

                if weighted_mse < best_mse_val:
                    best_mse_val = weighted_mse
                    best_split = (left, right)
                    best_col = col
                    best_col_val = split_value
                    best_linear_regressions = (left_lr, right_lr)
        return best_split, best_col, best_col_val, best_linear_regressions



    # Split a dataset based on an attribute and an attribute value
    def split_dataset_for_cal_val(self, col, value_to_split, dataset):
        left = dataset[dataset[col] < value_to_split]
        right = dataset[dataset[col] > value_to_split]
        return left, right

    def predict(self, x):
        return [self.find_leaf_node(row, self.root) for i, row in x.iterrows()]


    def find_leaf_node(self, x, node:TreeNode):
        if x[node.col] < node.col_val:
            if node.left is None:
                return node.linear_regression_left.predict([x])
            else:
                return self.find_leaf_node(x, node.left)
        else:
            if node.right is None:
                return node.linear_regression_right.predict([x])
            else:
                return self.find_leaf_node(x, node.right)





