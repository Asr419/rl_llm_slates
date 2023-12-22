import abc
from typing import TypeVar

import torch
import torch.nn as nn

torch_model = TypeVar("torch_model", bound=torch.nn.Module)


class SlateAgent(metaclass=abc.ABCMeta):
    # model an abstract agent recommending slates of documents
    def __init__(self, slate_gen) -> None:
        self.slate_gen = slate_gen

    def get_action(
        self, docs_scores: torch.Tensor, docs_qvalues: torch.Tensor
    ) -> torch.Tensor:
        """Get the action (slate) of the agent"""
        scores, ids = self.slate_gen(docs_scores, docs_qvalues)
        return ids
