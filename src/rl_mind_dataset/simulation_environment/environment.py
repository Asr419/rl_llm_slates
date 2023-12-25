import random as rand
from math import sqrt
from typing import Optional

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch


class SlateGym(gym.Env):
    def __init__(
        self,
        user_state,
        choice_model,
        response_model,
        num_candidates: int = 10,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.user_state = user_state
        self.device = device
        self.num_candidates = num_candidates
        self.choice_model = choice_model
        self.response_model = response_model

        # initialized by reset
        self.is_terminal = False

        self.curr_user = torch.Tensor
        self.candidate_docs: torch.Tensor
        self.clicked_docs: torch.Tensor

    def step(
        self,
        slate: torch.Tensor,
        iterator: int,
        cdocs_subset_idx: Optional[torch.Tensor] = None,
    ):
        if cdocs_subset_idx is not None:
            cdocs_feature = self.candidate_docs[cdocs_subset_idx, :]
        else:
            cdocs_feature = self.candidate_docs

        cdocs_feature = cdocs_feature[slate, :]

        # select from the slate on item following the user choice model
        hidden_state = self.user_state._generate_hidden_state()
        self.choice_model.score_documents(hidden_state, cdocs_feature)
        selected_doc_idx = self.choice_model.choose_document()

        if selected_doc_idx == self.choice_model.no_selection_token:
            # print("No document selected")
            response = self.response_model.generate_null_response()

            # create a fake selected_doc_feature of all zeros
            selected_doc_feature = torch.zeros(cdocs_feature.shape[1])
        else:
            # print("Document selected")
            # An item has been selected
            # TODO:remove, but can be userful for debugging
            doc_id = slate[selected_doc_idx]

            # checnum_candidates if user has selected a document
            selected_doc_feature = cdocs_feature[selected_doc_idx, :]

            # TODO: remove generate topic response and fix it in the response model
            response = self.response_model._generate_response(
                hidden_state,
                selected_doc_feature,
                self.clicked_docs[iterator],
            )

            # update user state
        next_user_state = self.user_state.update_state(
            selected_doc_feature=selected_doc_feature
        )

        if len(self.clicked_docs) == iterator:
            self.is_terminal = True
        info = {}

        return (
            selected_doc_feature,
            response,
            self.is_terminal,
            next_user_state,
            False,
            info,
        )

    def reset(self) -> None:
        # 1) sample user
        self.curr_user = self.user_state._generate_observable_state()
        self.is_terminal = False

    def render(self):
        raise NotImplementedError()

    def get_candidate_docs(self) -> torch.Tensor:
        self.candidate_docs = torch.stack(self.user_state._user_candidate_items())
        return self.candidate_docs

    def get_clicked_docs(self) -> torch.Tensor:
        self.clicked_docs = torch.stack(self.user_state.selected_item_feature())
        return self.clicked_docs
