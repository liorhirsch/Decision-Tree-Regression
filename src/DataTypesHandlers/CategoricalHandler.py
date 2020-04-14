from src.DataTypesHandlers.BaseHandler import BaseHandler
from src.TreeNodes.TreeNodeCategorical import TreeNodeCategorical


class CategoricalHandler(BaseHandler):
    def split(self, col):
        group_to_dataset = list(self.dataset.groupby(col))
        groups_to_lr_model = {}
        groups = []

        number_of_categories = sum([len(g[1]) >= self.min_elememnts_in_node for g in group_to_dataset])

        if (number_of_categories <= 1):
            return None, None, None

        weighted_mse = 0

        for g in group_to_dataset:
            group, group_dataset = g
            if len(group_dataset) < self.min_elememnts_in_node:
                continue

            model, mse = self.fit_lr_model(group_dataset)
            weighted_mse += len(group_dataset) * mse
            groups_to_lr_model[group] = model
            groups.append(g)

        weighted_mse /= len(self.dataset)

        return groups_to_lr_model, weighted_mse, groups

    def build_tree_node(self, groups_to_lr_model, group_to_dataset, col) -> TreeNodeCategorical:
        categories = []
        dataset_groups = []
        lr_models = []

        for curr_category, curr_dataset  in group_to_dataset:
            if (len(curr_dataset) > 0):
                categories.append(curr_category)
                dataset_groups.append(curr_dataset)
                lr_models.append(groups_to_lr_model[curr_category])

        return TreeNodeCategorical(categories, dataset_groups, lr_models, col)
