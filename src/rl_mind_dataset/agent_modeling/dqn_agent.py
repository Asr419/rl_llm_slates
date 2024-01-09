import random
import time
from collections import deque, namedtuple
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset

from rl_mind_dataset.agent_modeling.agent import SlateAgent


# defining different kind of named tuples for different agents
class Transition(NamedTuple):
    state: torch.Tensor
    selected_doc_feat: torch.Tensor
    candidate_docs: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor


class ReplayMemoryDataset(IterableDataset):
    def __init__(self, capacity: int, transition_cls):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.Transition = transition_cls

    def __iter__(self):
        random.shuffle(self.memory)
        return iter(self.memory)

    def push(self, transition):
        """Save a transition"""
        self.memory.append(transition)

    def collate_fn(self, batch):
        transitions_batch = self.Transition(*zip(*batch))
        preprocessed_batch = [
            torch.stack(getattr(transitions_batch, field_name))
            for field_name in transitions_batch._fields
        ]
        return preprocessed_batch


class DQNnet(nn.Module):
    def __init__(self, input_size, hidden_dims: list[int], output_size=1):
        # todo: change intra dimensions
        super(DQNnet, self).__init__()

        self.layers = nn.ModuleList()

        # create layers based on hidden_dims
        prev_dim = input_size
        for dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        # add last layer
        self.layers.append(nn.Linear(prev_dim, output_size))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.leaky_relu(x)
        return x


class DQNAgent(SlateAgent, nn.Module):
    def __init__(
        self,
        slate_gen,
        input_size: int,
        output_size: int,
        hidden_dims: list[int] = [30, 27, 24, 21, 18, 15, 12, 9, 6, 3],
        tau: float = 0.001,
    ) -> None:
        # init super classes
        SlateAgent.__init__(self, slate_gen=slate_gen)
        nn.Module.__init__(self)
        self.tau = tau
        # init DQN nets
        self.policy_net = DQNnet(
            input_size=input_size, output_size=output_size, hidden_dims=hidden_dims
        )
        # note that the target network is not updated during training
        self.target_net = DQNnet(
            input_size=input_size, output_size=output_size, hidden_dims=hidden_dims
        )
        self.target_net.requires_grad_(False)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update_target_network(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def compute_q_values(
        self,
        state: torch.Tensor,
        candidate_docs_repr: torch.Tensor,
        use_policy_net: bool = True,
    ) -> torch.Tensor:
        # concatenate state and candidate docs
        input1 = torch.cat([state, candidate_docs_repr], dim=1)
        # [num_candidate_docs, 1]
        if use_policy_net:
            q_val = self.policy_net(input1)
        else:
            q_val = self.target_net(input1)
        return q_val
