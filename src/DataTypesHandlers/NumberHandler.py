from src.DataTypesHandlers.BaseHandler import BaseHandler
import numpy as np

from src.TreeNodes.BaseTreeNode import BaseTreeNode
from src.TreeNodes.TreeNodeNumerical import TreeNodeNumerical


class NumberHandler(BaseHandler):
    def split(self, col):
        best_mse_val = np.inf
        chosen_linear_regression = []
        best_split = []
        chosen_col_val = None

        for index, row in self.dataset.iterrows():
            split_value = row[col]

            left_dataset, right_dataset = self.split_dataset_by_col_val(col, split_value)

            if (len(left_dataset) < self.min_elememnts_in_node or
                    len(right_dataset) < self.min_elememnts_in_node):
                continue

            left_lr, left_mse = self.fit_lr_model(left_dataset)
            right_lr, right_mse = self.fit_lr_model(right_dataset)

            weighted_mse = (len(right_dataset) * right_mse + len(left_dataset) * left_mse) / len(self.dataset)

            if weighted_mse < best_mse_val:
                best_mse_val = weighted_mse
                best_split = [left_dataset, right_dataset]
                chosen_col_val = split_value
                chosen_linear_regression = [left_lr, right_lr]
        return chosen_linear_regression, best_mse_val, best_split, chosen_col_val

    def split_dataset_by_col_val(self, col, value_to_split):
        left = self.dataset[self.dataset[col] < value_to_split]
        right = self.dataset[self.dataset[col] > value_to_split]
        return left, right

    def build_tree_node(self, value, dataset_groups, lr_models, col) -> TreeNodeNumerical:
        return TreeNodeNumerical(value, dataset_groups, lr_models, col)


