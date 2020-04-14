from sklearn.linear_model._base import LinearModel

from src.TreeNodes.BaseTreeNode import BaseTreeNode


class TreeNodeNumerical(BaseTreeNode):
    def get_match_lr_child(self, row):
        if row[self.col] < self.value:
            return self.lr_models[0], self.children_nodes[0]
        else:
            return self.lr_models[1], self.children_nodes[1]