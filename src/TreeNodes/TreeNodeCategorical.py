from src.TreeNodes.BaseTreeNode import BaseTreeNode
import numpy as np


class TreeNodeCategorical(BaseTreeNode):
    def get_match_lr_child(self, row):
        row_category = row[self.col]
        if row_category in self.value:
            idx = self.value.index(row_category)

            return self.lr_models[idx], self.children_nodes[idx], self.dataset_groups[idx]

        return None, None, None