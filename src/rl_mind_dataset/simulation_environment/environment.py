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
        self.hidden_choice_state = torch.Tensor
        self.candidate_docs: torch.Tensor
        self.clicked_docs: torch.Tensor
        self.diversity_value: torch.Tensor

    def step(
        self,
        slate: torch.Tensor,
        iterator: int,
        cdocs_subset_idx: Optional[torch.Tensor] = None,
        test=False,
    ):
        if cdocs_subset_idx is not None:
            cdocs_subset_idx = cdocs_subset_idx.to(self.device)
            self.candidate_docs = self.candidate_docs.to(self.device)
            cdocs_feature = self.candidate_docs[cdocs_subset_idx, :].to(self.device)
        else:
            cdocs_feature = self.candidate_docs.to(self.device)

        cdocs_feature = cdocs_feature[slate, :].to(self.device)
        # diverse_topics = torch.sum(torch.sum(cdocs_feature, dim=0) > 0)
        diverse_score = self.user_state.categorical_diversity(cdocs_feature)
        # diverse_curr_user = self.curr_user.view(1, -1).to(self.device)

        # # Calculate cosine similarity
        # cosine_similarities = torch.nn.functional.cosine_similarity(
        #     diverse_curr_user, cdocs_feature, dim=1
        # )

        # # Calculate diverse score (sum of cosine similarities)
        # diverse_score = torch.sum(cosine_similarities) / 5

        # select from the slate on item following the user choice model
        # hidden_state = self.user_state._generate_hidden_state()
        hidden_step_state = self.clicked_docs[iterator].to(self.device)
        hidden_step_state_rep = hidden_step_state.repeat(
            (cdocs_feature.shape[0], 1)
        ).to(self.device)
        self.choice_model.score_documents(hidden_step_state_rep, cdocs_feature)
        selected_doc_idx = self.choice_model.choose_document()

        if selected_doc_idx == self.choice_model.no_selection_token:
            # print("No document selected")
            response = self.response_model.generate_null_response()
            user_satisfaction = torch.tensor(0)
            relevance = torch.tensor(0)

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

            response, user_satisfaction, relevance = (
                self.response_model._generate_response(
                    self.hidden_choice_state.to(self.device),
                    selected_doc_feature.to(self.device),
                    self.clicked_docs[iterator].to(self.device),
                    diversity=diverse_score,
                    alpha=self.diversity_value,
                )
            )
        if test:
            next_user_state = torch.zeros(50).to(self.device)
        else:
            # update user state
            next_user_state = self.user_state.update_state(
                selected_doc_feature=selected_doc_feature.to(self.device)
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
            diverse_score,
            user_satisfaction,
            relevance,
        )

    def reset(self) -> None:
        # 1) sample user
        self.curr_user = self.user_state._generate_observable_state()
        self.is_terminal = False

    def render(self):
        raise NotImplementedError()

    def get_candidate_docs(self) -> torch.Tensor:
        self.candidate_docs = torch.stack(self.user_state._user_candidate_items())
        return self.candidate_docs.to(self.device)

    def get_clicked_docs(self) -> torch.Tensor:
        self.clicked_docs = torch.stack(self.user_state.selected_item_feature())
        return self.clicked_docs

    def hidden_state(self):
        self.hidden_choice_state = self.user_state._generate_hidden_state()
        return self.hidden_choice_state

    def diversity(self):
        self.diversity_value = self.user_state.entropy_based_diversity()
        return self.diversity_value
