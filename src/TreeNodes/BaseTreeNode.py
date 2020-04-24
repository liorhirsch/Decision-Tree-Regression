from typing import List

import pandas as pd
from sklearn.linear_model._base import LinearModel


class BaseTreeNode():

    dataset_groups: List[pd.DataFrame]
    lr_models: List[LinearModel]
    children_nodes: List['BaseTreeNode']
    self_lr_model: LinearModel

    def __init__(self, value, dataset_groups, lr_models, col) -> None:
        self.value = value
        self.dataset_groups = dataset_groups
        self.lr_models = lr_models
        self.col = col
        self.children_nodes = []

    def get_match_lr_child(self, row):
        return self.lr_models[0], None
