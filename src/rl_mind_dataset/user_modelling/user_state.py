import abc
import os
from pathlib import Path
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

    @abc.abstractmethod
    def update_state(self, selected_doc_feature: torch.Tensor) -> None:
        """Update the user's observable state"""
        pass


class UserState(AbstractUserState):
    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.DATA_PATH = Path.home() / Path(os.environ.get("DATA_PATH"))
        self.dataset_reader = DatasetReader()

        self.dataset_interaction_path = self.DATA_PATH / Path(
            "MINDlarge_train/interaction.feather"
        )
        self.interaction_data = pd.read_feather(self.dataset_interaction_path)
        self.dataset_path = self.DATA_PATH / Path("MINDlarge_train/category.feather")
        self.category_data = pd.read_feather(self.dataset_path)
        self.user_state = self._generate_observable_state()
        device: torch.device = (torch.device("cpu"),)

    def _generate_observable_state(self, **kwds: Any) -> torch.Tensor:
        num_rows = len(self.category_data)
        random_index = np.random.randint(0, num_rows)
        user_state = torch.Tensor(self.category_data["category_list"].loc[random_index])
        print("user_state generated")
        return user_state

    def _generate_hidden_state(self, **kwds: Any) -> Tensor:
        normalized_user_state = self.user_state / self.user_state.sum()
        sampled_value = torch.multinomial(normalized_user_state, 1).item()
        self.hidden_user_state = torch.zeros_like(
            normalized_user_state, dtype=torch.float32
        )
        self.hidden_user_state[sampled_value] = 1
        return self.hidden_user_state

    def update_state(self, selected_doc_feature: torch.Tensor) -> None:
        """Update the user's observable state"""
        pass
