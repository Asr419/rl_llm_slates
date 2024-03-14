import abc
import os
from pathlib import Path
import random
from typing import Any, Type, TypeVar


from rl_mind_dataset.document_modelling.data_reader import DatasetReader

import pandas as pd
import numpy as np
import numpy.typing as npt
import torch
from torch._tensor import Tensor
import torch.nn as nn


class AbstractUserState(nn.Module, metaclass=abc.ABCMeta):
    # hidden state of the user
    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.dataset = None
        self.user_state = None

    @abc.abstractmethod
    def _generate_observable_state(self, **kwds: Any) -> torch.Tensor:
        """Generate the user observable state from click history data"""
        pass

    def _generate_hidden_state(self, **kwds: Any) -> torch.Tensor:
        """Generate the user hidden state by sampling a topic of interest"""
        pass

    def _user_candidate_items(self, **kwds: Any) -> torch.Tensor:
        """Generate the user candidate items"""
        pass

    @abc.abstractmethod
    def update_state(self, selected_doc_feature: torch.Tensor) -> None:
        """Update the user's observable state"""
        pass


class UserState(AbstractUserState):
    def __init__(
        self, device: torch.device = torch.device("cpu"), test=False, **kwds: Any
    ) -> None:
        super().__init__(**kwds)
        self.DATA_PATH = Path.home() / Path(os.environ.get("DATA_PATH"))
        self.dataset_reader = DatasetReader()
        self.random_index = None
        self.user_state: Tensor = None
        self.clicked_items = []
        self.item_tensor: Tensor = None
        self.device = device
        if test:
            self.dataset_interaction_path = self.DATA_PATH / Path(
                "MINDlarge_train/test_50.feather"
            )
            self.interaction_data = pd.read_feather(self.dataset_interaction_path)
            self.dataset_path = self.DATA_PATH / Path("MINDlarge_train/test_50.feather")
        else:
            self.dataset_interaction_path = self.DATA_PATH / Path(
                "MINDlarge_train/interaction_all_50.feather"
            )
            self.interaction_data = pd.read_feather(self.dataset_interaction_path)
            self.dataset_path = self.DATA_PATH / Path(
                "MINDlarge_train/interaction_all_50.feather"
            )
        self.category_data = pd.read_feather(self.dataset_path)
        device: torch.device = (torch.device("cpu"),)
        self.embedding_dict, self.all_item_vectors = self.dataset_reader.item2vecdict()
        self.article_category = self.dataset_reader.article_category_map()
        self.new_dict = {
            tuple(value): key for key, value in self.embedding_dict.items()
        }

    # def _generate_observable_state(self, **kwds: Any) -> torch.Tensor:
    #     num_rows = len(self.category_data)
    #     self.random_index = np.random.randint(0, num_rows)

    #     numpy_array = np.copy(
    #         self.category_data["category_list"].loc[self.random_index]
    #     )

    #     if numpy_array.size == 0:
    #         self.user_state = torch.randint(2, size=(18,), dtype=torch.float32)
    #     else:
    #         self.user_state = torch.Tensor(numpy_array)

    #     return self.user_state
    def _generate_observable_state(self, **kwds: Any) -> torch.Tensor:
        num_rows = len(self.category_data)
        self.random_index = np.random.randint(0, num_rows)

        numpy_array = np.copy(
            self.category_data["observed_state"].loc[self.random_index]
        )
        numpy_array = numpy_array.astype(float)
        if numpy_array.size == 0 or np.isnan(numpy_array).any():
            self.user_state = 2 * torch.rand(50, dtype=torch.float32) - 1
        else:
            self.user_state = torch.Tensor(numpy_array.astype(float))

        return self.user_state

    # def _user_candidate_items(self, **kwds: Any) -> Tensor:
    #     items = self.category_data["presented_slate"].loc[self.random_index]
    #     embedding_dict = self.dataset_reader.item2vecdict()
    #     item_list = [embedding_dict.get(key, []) for key in items]

    #     remaining_items = 150 - len(item_list)
    #     # additional_values = random.choices(
    #     #     list(set(range(18)) - set(item_list)), k=remaining_items
    #     # )
    #     additional_values = random.choices(list(set(range(18))), k=remaining_items)
    #     item_list.extend(additional_values)
    #     random.shuffle(item_list)
    #     # length_of_list = len(item_list)
    #     candidate_tensor = [torch.eye(18)[value] for value in item_list]
    #     return candidate_tensor
    def _user_candidate_items(self, **kwds: Any) -> Tensor:
        items = self.category_data["presented_slate"].loc[self.random_index]
        item_list = [
            self.embedding_dict.get(key, [])
            for key in items
            if self.embedding_dict.get(key, []) is not None
            and len(self.embedding_dict.get(key, [])) > 0
        ]
        # set the candidate documents
        remaining_items = 100 - len(item_list)
        # item_list_arrays = [np.array(vector) for vector in item_list]
        # item_list_set = {tuple(item) for item in item_list_arrays}
        # available_vectors = [
        #     vector for vector in all_item_vectors if vector not in item_list_set
        # ]
        if remaining_items > 0:
            available_items = [
                sublist
                for sublist in self.all_item_vectors
                if not any(
                    np.array_equal(sublist, item) and np.size(sublist) > 2
                    for item in item_list
                )
            ]
            # item_list_arrays = [np.array(vector) for vector in item_list]
            # available_vectors = [
            #     vector
            #     for vector in self.all_item_vectors
            #     if not any(
            #         np.array_equal(vector, item) and np.size(vector) >= 2
            #         for item in item_list_arrays
            #     )
            # ]
            selected_lists = random.sample(
                available_items, min(remaining_items, len(available_items))
            )
            item_list.extend(selected_lists)
        # make sure you take only the subset
        item_list = item_list[:100]
        random.shuffle(item_list)
        candidate_tensor = [torch.Tensor(value.astype(float)) for value in item_list]
        # items = self.category_data["presented_slate"].loc[self.random_index]
        # embedding_dict, all_item_vectors = self.dataset_reader.item2vecdict()
        # item_list = [embedding_dict.get(key, []) for key in items]
        # item_list = [lst for lst in item_list if len(lst) > 0]
        # remaining_items = 150 - len(item_list)
        # item_list_arrays = [np.array(vector) for vector in item_list]
        # available_vectors = [
        #     vector
        #     for vector in all_item_vectors
        #     if not any(np.array_equal(vector, item) for item in item_list_arrays)
        # ]
        # selected_lists = random.sample(available_vectors, remaining_items)
        # item_list.extend(selected_lists)
        # random.shuffle(item_list)
        # candidate_tensor = [torch.Tensor(value) for value in item_list]
        return candidate_tensor

    def _user_selected_item(self, **kwds: Any) -> Tensor:
        user_id = self.category_data["userId"].loc[self.random_index]
        timestamp = self.category_data["timestamp"].loc[self.random_index]
        self.clicked_items = self.category_data[
            (self.category_data["userId"] == user_id)
            & (self.category_data["timestamp"] == timestamp)
        ]["click"].tolist()
        clicked_items = self.clicked_items
        return clicked_items

    def selected_item_feature(self, **kwds: Any) -> Tensor:
        selected_items = self._user_selected_item()
        selected_item_list = [
            self.embedding_dict.get(key, [])
            for key in selected_items
            if len(self.embedding_dict.get(key, [])) > 0
        ]
        selected_candidate_tensor = [
            torch.Tensor(value) for value in selected_item_list
        ]
        return selected_candidate_tensor

    def _generate_hidden_state(
        self,
        **kwds: Any,
    ) -> Tensor:
        sum_tensor = torch.sum(torch.stack(self.selected_item_feature()), dim=0)
        self.hidden_user_state = sum_tensor / torch.norm(sum_tensor)

        return self.hidden_user_state
        # normalized_user_state = self.user_state / self.user_state.sum()
        # sampled_value = torch.multinomial(normalized_user_state, 1).item()
        # self.hidden_user_state = torch.zeros_like(
        #     normalized_user_state, dtype=torch.float32
        # )
        # self.hidden_user_state[sampled_value] = 1
        # return self.hidden_user_state

    def diversity_coefficient(self, **kwds: Any) -> Tensor:
        # calculate the diversity coefficient between self.user_state and self.item_tensor
        diversity_coefficient = torch.tensor(0.0)
        items_hist = self.category_data["click_history"].loc[self.random_index]
        item_list_hist = [
            self.embedding_dict.get(key, [])
            for key in items_hist
            if self.embedding_dict.get(key, []) is not None
            and len(self.embedding_dict.get(key, [])) > 0
        ]
        self.item_tensor = [
            torch.tensor(array, dtype=torch.float) for array in item_list_hist
        ]
        if len(self.item_tensor) == 0:
            return diversity_coefficient
        else:
            item_tensor_cosine = torch.stack(self.item_tensor)
            user_state = self.user_state.unsqueeze(0).expand_as(item_tensor_cosine)
            cosine_similarities = torch.nn.functional.cosine_similarity(
                user_state, item_tensor_cosine, dim=1
            )
            diversity_coefficient = torch.mean(cosine_similarities)
            return diversity_coefficient

    def update_state(
        self, selected_doc_feature: torch.Tensor, sigma: int = 1.0
    ) -> None:
        # self.user_state=(self.user_state+selected_doc_feature)/2
        self.user_state = self.user_state.to(self.device)
        distance_squared = torch.sum((self.user_state - selected_doc_feature) ** 2)
        similarity = torch.exp(-distance_squared / (2 * sigma**2))
        # I = torch.dot(self.user_state, selected_doc_feature)  # type: ignore
        p_positive = (similarity + 1) / 2

        if torch.rand(1).to(self.device) < p_positive:  #
            self.user_state = self.user_state + (1 - similarity) * selected_doc_feature
        else:
            self.user_state = self.user_state - (1 - similarity) * selected_doc_feature
        # self.user_state = torch.clamp(self.user_state, min=-1.0, max=1.0)

        # self.user_state = torch.mean(
        #     torch.stack([self.user_state, selected_doc_feature]), dim=0
        # )
        return self.user_state
        """Update the user's observable state"""

    def diversity_dissimilarity(self, **kwds: Any) -> Tensor:
        items_hist = self.category_data["click_history"].loc[self.random_index]
        item_list_hist = [
            self.embedding_dict.get(key, [])
            for key in items_hist
            if self.embedding_dict.get(key, []) is not None
            and len(self.embedding_dict.get(key, [])) > 0
        ]
        self.item_tensor = [
            torch.tensor(array, dtype=torch.float) for array in item_list_hist
        ]
        if len(self.item_tensor) >= 2:
            tensors = self.item_tensor
            n = len(tensors)
            similarity_matrix = torch.zeros(n, n)

            # Compute similarity matrix
            for i in range(n):
                for j in range(i + 1, n):
                    similarity_matrix[i, j] = torch.dot(tensors[i], tensors[j]) / (
                        torch.norm(tensors[i]) * torch.norm(tensors[j])
                    )
                    similarity_matrix[j, i] = similarity_matrix[
                        i, j
                    ]  # Similarity matrix is symmetric

            # Calculate diversity
            total_diversity = (
                torch.sum(1 - similarity_matrix) - n
            )  # Exclude diagonal elements
            diversity_measure = total_diversity / ((n / 2) * (n - 1))
        else:
            diversity_measure = torch.tensor(0.0)

        return diversity_measure

    def categorical_diversity(self, slate_items, **kwds: Any) -> Tensor:

        item_ids = [self.new_dict.get(tuple(key.tolist()), 0) for key in slate_items]

        categories = [
            self.article_category.get(article_id, 0) for article_id in item_ids
        ]
        count_categories = [categories.count(i) for i in range(0, 18)]
        score = sum(1 for x in count_categories if x > 0)
        return score

    def click_history_diversity(self, **kwds: Any) -> Tensor:
        items_hist = self.category_data["click_history"].loc[self.random_index]
        categories = [
            self.article_category.get(article_id, 0) for article_id in items_hist
        ]
        count_categories = [categories.count(i) for i in range(0, 18)]
        score = sum(1 for x in count_categories if x > 0)
        a = 0
        return score / 18
