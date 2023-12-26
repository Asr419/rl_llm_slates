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
    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.DATA_PATH = Path.home() / Path(os.environ.get("DATA_PATH"))
        self.dataset_reader = DatasetReader()
        self.random_index = None
        self.user_state: Tensor = None
        self.clicked_items = []

        self.dataset_interaction_path = self.DATA_PATH / Path(
            "MINDlarge_train/interaction.feather"
        )
        self.interaction_data = pd.read_feather(self.dataset_interaction_path)
        self.dataset_path = self.DATA_PATH / Path("MINDlarge_train/category.feather")
        self.category_data = pd.read_feather(self.dataset_path)
        device: torch.device = (torch.device("cpu"),)

    def _generate_observable_state(self, **kwds: Any) -> torch.Tensor:
        num_rows = len(self.category_data)
        self.random_index = np.random.randint(0, num_rows)

        numpy_array = np.copy(
            self.category_data["category_list"].loc[self.random_index]
        )

        if numpy_array.size == 0:
            self.user_state = torch.randint(2, size=(18,), dtype=torch.float32)
        else:
            self.user_state = torch.Tensor(numpy_array)

        return self.user_state

    def _user_candidate_items(self, **kwds: Any) -> Tensor:
        items = self.category_data["presented_slate"].loc[self.random_index]
        article_category_map = self.dataset_reader.article_category_map()
        item_list = [article_category_map.get(key, []) for key in items]

        remaining_items = 100 - len(item_list)
        # additional_values = random.choices(
        #     list(set(range(18)) - set(item_list)), k=remaining_items
        # )
        additional_values = random.choices(list(set(range(18))), k=remaining_items)
        item_list.extend(additional_values)
        random.shuffle(item_list)
        # length_of_list = len(item_list)
        candidate_tensor = [torch.eye(18)[value] for value in item_list]
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
        selected_item = self._user_selected_item()
        article_category_map = self.dataset_reader.article_category_map()
        selected_item_list = [
            article_category_map.get(key, []) for key in self.clicked_items
        ]
        selected_candidate_tensor = [
            torch.eye(18)[value] for value in selected_item_list
        ]
        return selected_candidate_tensor

    def _generate_hidden_state(self, **kwds: Any) -> Tensor:
        normalized_user_state = self.user_state / self.user_state.sum()
        sampled_value = torch.multinomial(normalized_user_state, 1).item()
        self.hidden_user_state = torch.zeros_like(
            normalized_user_state, dtype=torch.float32
        )
        self.hidden_user_state[sampled_value] = 1
        return self.hidden_user_state

    def update_state(self, selected_doc_feature: torch.Tensor) -> None:
        topic_position = selected_doc_feature.argmax()
        if not torch.all(selected_doc_feature == 0):
            self.user_state[topic_position] += 1
        return self.user_state
        """Update the user's observable state"""
