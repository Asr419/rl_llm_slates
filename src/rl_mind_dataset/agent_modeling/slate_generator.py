import abc
import itertools
import random
from typing import Tuple

import numpy as np
import torch

from rl_mind_dataset.agent_modeling.dqn_agent import DQNnet


class AbstractSlateGenerator(metaclass=abc.ABCMeta):
    def __init__(self, slate_size: int = 10) -> None:
        self.slate_size = slate_size

    @abc.abstractmethod
    def __call__(
        self,
        docs_id: torch.Tensor,
        docs_scores: torch.Tensor,
        docs_qvalues: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a state and a set of candidate documents, create a slate of documents"""
        pass


class TopKSlateGenerator(AbstractSlateGenerator):
    def __call__(
        self, docs_scores: torch.Tensor, docs_qvalues: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        topk_scores, topk_ids = torch.topk(
            docs_scores * docs_qvalues, k=self.slate_size
        )
        return topk_scores, topk_ids


class RandomSlateGenerator(AbstractSlateGenerator):
    def __call__(self, docs_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        slate_ids = torch.randint(
            low=0, high=len(docs_scores) - 1, size=(self.slate_size,)
        )
        slate_scores = torch.rand((self.slate_size,))
        return slate_scores, slate_ids


class TopKGreedySlateGenerator(AbstractSlateGenerator):
    def __call__(
        self,
        docs_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        topk_scores, topk_ids = torch.topk(docs_scores, k=self.slate_size)
        return topk_scores, topk_ids


# class DiverseSlateGenerator(AbstractSlateGenerator):
#     def __call__(
#         self,
#         docs_scores: torch.Tensor,
#         docs_qvalues: torch.Tensor,
#         candidate_docs_repr: torch.Tensor,
#     ) -> Tuple[torch.Tensor]:
#         # calculate the variance of each feature
#         variances_tensor = torch.var(candidate_docs_repr, dim=0)

#         # calculate the diversity score for each song
#         mean_features_tensor = torch.mean(candidate_docs_repr, dim=0)
#         diversity_scores_tensor = torch.sqrt(
#             torch.sum((candidate_docs_repr - mean_features_tensor) ** 2, dim=1)
#         ) * torch.sum(variances_tensor)

#         # combine the scores and diversity into a single tensor
#         scores_tensor = torch.stack(
#             [docs_scores, docs_qvalues, diversity_scores_tensor], dim=1
#         )

#         # sort the songs by their scores and diversity
#         topk_scores, topk_ids = torch.topk(scores_tensor, k=self.slate_size)
#         # sorted_indices_tensor = torch.argsort(scores_tensor, dim=0, descending=True)

#         # # select the top k songs based on scores and diversity
#         # k = self.slate_size
#         # top_songs_indices_tensor = sorted_indices_tensor[:k]
#         # top_songs_tensor = candidate_docs_repr[top_songs_indices_tensor]

#         return topk_scores, topk_ids


# # todo: to be checked
# class GreedySlateGenerator(AbstractSlateGenerator):
#     def __call__(
#         self,
#         docs_scores: torch.Tensor,
#         docs_qvals: torch.Tensor,
#     ) -> Tuple[torch.Tensor]:
#         # (slate_size, s_no_click, s, q):
#         def argmax(v, mask):
#             return torch.argmax((v - torch.min(v)) * mask, dim=0)

#         numerator = torch.tensor(0.0)
#         denominator = torch.tensor(-1.0)  # set s_no_click to -1.0
#         mask = torch.ones_like(docs_qvals)

#         def set_element(v, i, x):
#             mask = torch.zeros_like(v)
#             mask[i] = 1
#             v_new = torch.ones_like(v) * x
#             return torch.where(mask == 1, v_new, v)

#         for _ in range(self.slate_size):
#             k = argmax(
#                 (numerator + docs_scores * docs_qvals) / (denominator + docs_scores),
#                 mask,
#             )
#             mask = set_element(mask, k, 0)
#             numerator = numerator + docs_scores * docs_qvals[k]
#             denominator = denominator + docs_scores[k]

#         output_slate = torch.where(mask == 0)[0]
#         return output_slate


# class OptimalSlateGenerator(AbstractSlateGenerator):
#     def __call__(
#         self,
#         docs_scores: torch.Tensor,
#         docs_qvals: torch.Tensor,
#     ) -> Tuple[torch.Tensor]:
#         num_candidates = docs_scores.shape[0]
#         s_no_click = torch.tensor(-1.0)
#         num_samples = 10000

#         slates = torch.rand((num_samples, self.slate_size)) * num_candidates
#         slates = slates.floor().to(torch.long)

#         # Compute the slate scores and q values.
#         # Docs not in slate get a score of 0 and q value of 1.
#         slate_scores = torch.zeros((num_samples, self.slate_size))
#         slate_qvals = torch.ones((num_samples, self.slate_size))
#         for i in range(self.slate_size):
#             idx = slates[:, i]
#             slate_scores[:, i] = docs_scores[idx]
#             slate_qvals[:, i] = docs_qvals[idx]

#         # Compute the total score and q value for each slate.
#         slate_q_values = slate_scores * slate_qvals
#         slate_normalizer = slate_scores.sum(dim=1) + s_no_click
#         slate_q_values = slate_q_values / slate_normalizer.unsqueeze(1)
#         slate_sum_q_values = slate_q_values.sum(dim=1)

#         # Select the slate with the highest expected reward.
#         max_q_slate_index = slate_sum_q_values.argmax()

#         return slates[max_q_slate_index]
