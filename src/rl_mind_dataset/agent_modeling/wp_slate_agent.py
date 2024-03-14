import abc
from typing import Tuple
import time
import numpy as np
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WolpertingerActorSlate(nn.Module):
    def __init__(
        self, nn_dim: list[int], k: int, input_dim: int = 50, slate_size: int = 10
    ):
        super(WolpertingerActorSlate, self).__init__()
        self.k = k

        layers = []
        for i, dim in enumerate(nn_dim):
            if i == 0:
                layers.append(nn.Linear(input_dim, dim))
            elif i == len(nn_dim) - 1:
                layers.append(nn.Linear(nn_dim[i - 1], slate_size * 50))
            else:
                layers.append(nn.Linear(nn_dim[i - 1], dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # output protoaction
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
        return x


class ActorAgentSlate(nn.Module):
    def __init__(
        self,
        nn_dim: list[int],
        k: int,
        input_dim: int = 50,
        tau: float = 0.001,
        slate_size: int = 10,
    ) -> None:
        nn.Module.__init__(self)
        self.tau = tau
        self.actor_policy_net = WolpertingerActorSlate(
            nn_dim=nn_dim, k=k, input_dim=input_dim, slate_size=slate_size
        )
        self.actor_target_net = WolpertingerActorSlate(
            nn_dim=nn_dim, k=k, input_dim=input_dim, slate_size=slate_size
        )
        self.actor_target_net.requires_grad_(False)
        self.actor_target_net.load_state_dict(self.actor_policy_net.state_dict())
        self.k = k

    def soft_update_target_network(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.actor_target_net.state_dict()
        policy_net_state_dict = self.actor_policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.actor_target_net.load_state_dict(target_net_state_dict)

    def compute_proto_slate(
        self, state: torch.Tensor, use_actor_policy_net: bool = True
    ):
        if use_actor_policy_net:
            return self.actor_policy_net(state)
        else:
            return self.actor_target_net(state)

    def k_nearest(
        self,
        input_state: torch.Tensor,
        candidate_docs: torch.Tensor,
        use_actor_policy_net,
        slate_size: int = 5,
    ) -> None:
        proto_action = self.compute_proto_slate(
            input_state, use_actor_policy_net=use_actor_policy_net
        )
        proto_slate = proto_action.view(slate_size, 50)
        start = time.time()
        for count, i in enumerate(proto_slate):
            distances = torch.linalg.norm(candidate_docs - i, axis=1)
            # Sort distances and get indices of k smallest distances
            indices = torch.argsort(distances, dim=0)
            # candidates_subset = candidate_docs[indices]

            # # Select k closest tensors from tensor list

            # if count == 0:
            #     indices_tensor = indices
            #     candidates_tensor = candidates_subset
            # else:
            #     indices_tensor = torch.cat((indices_tensor, indices), dim=0)
            #     candidates_tensor = torch.cat(
            #         (candidates_tensor, candidates_subset), dim=0
            #     )

            if count == 0:
                indices_tensor = indices[: self.k]
                candidates_subset = candidate_docs[indices_tensor]
                candidates_tensor = candidates_subset

            else:
                remaining_indices = indices[
                    torch.logical_not(torch.isin(indices, indices_tensor))
                ][: self.k]
                indices_tensor = torch.cat((indices_tensor, remaining_indices), dim=0)
                candidates_subset = candidate_docs[remaining_indices]
                candidates_tensor = torch.cat(
                    (candidates_tensor, candidates_subset), dim=0
                )

        end = time.time()
        # print("k_nearest_time", end - start)
        # append indices to another tensor at each iteration

        # distances = torch.linalg.norm(candidate_docs - proto_action, axis=1)
        # # Sort distances and get indices of k smallest distances
        # indices = torch.argsort(distances, dim=0)[: self.k]
        # # Select k closest tensors from tensor list
        # candidates_subset = candidate_docs[indices]

        return candidates_tensor, indices_tensor

    def test_k_nearest(
        self,
        input_state: torch.Tensor,
        candidate_docs: torch.Tensor,
        use_actor_policy_net,
        nearest_neighbours,
    ) -> None:
        proto_action = self.compute_proto_action(
            input_state, use_actor_policy_net=use_actor_policy_net
        )
        distances = torch.linalg.norm(candidate_docs - proto_action, axis=1)
        # Sort distances and get indices of k smallest distances
        indices = torch.argsort(distances, dim=0)[:nearest_neighbours]
        # Select k closest tensors from tensor list
        candidates_subset = candidate_docs[indices]

        return candidates_subset, indices

    def k_nearest_to_state(
        self,
        input_state: torch.Tensor,
        candidate_docs: torch.Tensor,
    ) -> None:
        distances = torch.linalg.norm(candidate_docs - input_state, axis=1)
        # Sort distances and get indices of k smallest distances
        indices = torch.argsort(distances, dim=0)[: self.k]
        # Select k closest tensors from tensor list
        candidates_subset = candidate_docs[indices]

        return candidates_subset, indices
