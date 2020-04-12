from sklearn.linear_model._base import LinearModel


class TreeNode():
    left: 'TreeNode'
    right: 'TreeNode'
    linear_regression_left: LinearModel
    linear_regression_right: LinearModel

    def __init__(self, col, col_val, linear_regression_left, linear_regression_right):
        self.col = col
        self.col_val = col_val
        self.linear_regression_left = linear_regression_left
        self.linear_regression_right = linear_regression_right
