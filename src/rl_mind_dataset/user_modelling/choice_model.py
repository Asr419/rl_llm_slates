# implement user choice model
import abc
from dataclasses import dataclass
from typing import Any, Callable, Type, TypeVar
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from rl_mind_dataset.user_modelling.ncf import NCF, DataFrameDataset


# maybe the scoring function can be passed as a parameter
class AbstractChoiceModel(metaclass=abc.ABCMeta):
    _scores = None

    @property
    def scores(self):
        return self._scores

    @abc.abstractmethod
    def _score_documents(
        self, user_state: torch.Tensor, doc_repr: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def score_documents(
        self, user_state: torch.Tensor, docs_repr: torch.Tensor
    ) -> None:
        pass

    @abc.abstractmethod
    def choose_document(self, satisfaction_threshold) -> int:
        # return the index of the chosen document in the slate
        pass


class NormalizableChoiceModel(AbstractChoiceModel):
    """A normalizable choice model."""

    def __init__(
        self, satisfaction_threshold: float = 0.2, no_selection_token: int = -1
    ) -> None:
        self.satisfaction_threshold = satisfaction_threshold
        self.no_selection_token = no_selection_token

    def choose_document(self) -> int:
        assert (
            self._scores is not None
        ), "Scores are not computed yet. call score_documents() first."

        all_probs = self._scores
        # add null document, by adding a score of 0 at the last postin of the tensor _scores
        all_probs = torch.cat((all_probs, torch.tensor([-1.0])))
        all_probs = torch.softmax(all_probs, dim=0)
        # select the item according to the probability distribution all_probs
        selected_index = int(torch.multinomial(all_probs, num_samples=1).item())
        # check if the selected item is the null document return no_selection_token
        if selected_index == len(all_probs) - 1:
            return self.no_selection_token
        else:
            return selected_index

    @abc.abstractmethod
    def _score_documents(
        self, user_state: torch.Tensor, docs_repr: torch.Tensor
    ) -> torch.Tensor:
        pass

    def score_documents(self, user_state: torch.Tensor, docs_repr: torch.Tensor):
        self._scores = self._score_documents(user_state, docs_repr)


class DotProductChoiceModel(NormalizableChoiceModel):
    """A multinomial logit choice model with dot product as the scoring function."""

    def __init__(
        self,
        satisfaction_threshold: float = 0.0,
        no_selection_token: int = -1,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(satisfaction_threshold, no_selection_token)
        self.device = device

    def choose_document(self) -> int:
        assert (
            self._scores is not None
        ), "Scores are not computed yet. call score_documents() first."

        all_probs = self._scores.to(self.device)
        # add null document, by adding a score of 0 at the last postin of the tensor _scores
        all_probs = torch.cat((all_probs, torch.tensor([-1.0]).to(self.device)))
        all_probs = torch.softmax(all_probs, dim=0)
        # select the item according to the probability distribution all_probs
        selected_index = int(torch.multinomial(all_probs, num_samples=1).item())
        # check if the selected item is the null document return no_selection_token
        if selected_index == len(all_probs) - 1:
            return self.no_selection_token
        else:
            return selected_index

    def _score_documents(
        self, user_state: torch.Tensor, docs_repr: torch.Tensor
    ) -> torch.Tensor:
        # Calculate dot product between user_state and each document representation
        scores = torch.mm(user_state.unsqueeze(0), docs_repr.t()).squeeze(0)
        # normalize dot product values to 0 and 1 for convenience of training
        return scores

    def score_documents(self, user_state: torch.Tensor, docs_repr: torch.Tensor):
        self._scores = self._score_documents(user_state, docs_repr)


class CosineSimilarityChoiceModel(NormalizableChoiceModel):
    """A multinomial logit choice model with cosine similarity as the scoring function."""

    def __init__(
        self,
        satisfaction_threshold: float = 0.0,
        no_selection_token: int = -1,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(satisfaction_threshold, no_selection_token)
        self.device = device

    def choose_document(self) -> int:
        assert (
            self._scores is not None
        ), "Scores are not computed yet. call score_documents() first."

        all_probs = self._scores.to(self.device)
        # add null document, by adding a score of 0 at the last postin of the tensor _scores
        all_probs = torch.cat((all_probs, torch.tensor([-1.0]).to(self.device)))
        all_probs = torch.softmax(all_probs, dim=0)
        # select the item according to the probability distribution all_probs
        selected_index = int(torch.multinomial(all_probs, num_samples=1).item())
        # check if the selected item is the null document return no_selection_token
        if selected_index == len(all_probs) - 1:
            return self.no_selection_token
        else:
            return selected_index

    def _score_documents(
        self, user_state: torch.Tensor, docs_repr: torch.Tensor
    ) -> torch.Tensor:
        # Calculate cosine similarity between user_state and each document representation
        scores = F.cosine_similarity(user_state, docs_repr)
        scores = scores + 1
        # normalize cosine values to 0 and 1 for convenience of training
        # print(scores.max())
        return scores

    def score_documents(self, user_state: torch.Tensor, docs_repr: torch.Tensor):
        self._scores = self._score_documents(user_state, docs_repr)


class NCFChoiceModel(NormalizableChoiceModel):
    def __init__(
        self,
        satisfaction_threshold: float = 0.0,
        no_selection_token: int = -1,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(satisfaction_threshold, no_selection_token)
        self.device = device
        base_path = Path.home() / Path(os.environ.get("SAVE_PATH"))
        RUN_BASE_PATH = Path(f"user_choice_model")
        PATH = base_path / RUN_BASE_PATH / Path("model.pt")
        self.model = torch.load(PATH).to(self.device)

    def choose_document(self) -> int:
        assert (
            self._scores is not None
        ), "Scores are not computed yet. call score_documents() first."

        all_probs = self._scores.to(self.device)
        # add null document, by adding a score of 0 at the last postin of the tensor _scores
        all_probs = torch.cat((all_probs, torch.tensor([-1.0]).to(self.device)))
        all_probs = torch.softmax(all_probs, dim=0)
        # select the item according to the probability distribution all_probs
        selected_index = int(torch.multinomial(all_probs, num_samples=1).item())
        # check if the selected item is the null document return no_selection_token
        if selected_index == len(all_probs) - 1:
            return self.no_selection_token
        else:
            return selected_index

    def _score_documents(
        self, user_state: torch.Tensor, docs_repr: torch.Tensor
    ) -> torch.Tensor:
        # Calculate cosine similarity between user_state and each document representation
        scores = torch.sigmoid(self.model(user_state, docs_repr, device=self.device))
        # normalize cosine values to 0 and 1 for convenience of training
        # print(scores.max())
        return scores

    def score_documents(self, user_state: torch.Tensor, docs_repr: torch.Tensor):
        self._scores = self._score_documents(user_state, docs_repr)
