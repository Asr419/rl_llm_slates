import argparse
import configparser
import os
import pickle
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from rl_mind_dataset.agent_modeling.dqn_agent import (
    DQNAgent,
    ReplayMemoryDataset,
    Transition,
)
from rl_mind_dataset.agent_modeling.slate_generator import (  # DiverseSlateGenerator,; GreedySlateGenerator,; OptimalSlateGenerator,
    RandomSlateGenerator,
    TopKSlateGenerator,
    TopKGreedySlateGenerator,
)
from rl_mind_dataset.agent_modeling.wp_slate_agent import (
    WolpertingerActorSlate,
    ActorAgentSlate,
)
from rl_mind_dataset.agent_modeling.wp_agent import WolpertingerActor, ActorAgent

# from rl_mind_dataset.agent_modeling.wp_agent import WolpertingerActor, ActorAgent
# from rl_mind_dataset.agent_modeling.wp_slate_agent import (
#     WolpertingerActorSlate,
#     ActorAgentSlate,
# )
from rl_mind_dataset.document_modelling.data_reader import DatasetReader
from rl_mind_dataset.simulation_environment.environment import SlateGym
from rl_mind_dataset.user_modelling.choice_model import (
    CosineSimilarityChoiceModel,
    DotProductChoiceModel,
    NCFChoiceModel,
)

from rl_mind_dataset.user_modelling.response_model import (
    CosineResponseModel,
    DotProductResponseModel,
    WeightedDotProductResponseModel,
    WeightedCosineResponseModel,
)
from rl_mind_dataset.user_modelling.user_state import UserState


from rl_mind_dataset.utils import save_run, save_run_wa

class_name_to_class = {
    "ObservedUserState": UserState,
    "DotProductChoiceModel": DotProductChoiceModel,
    "NCFChoiceModel": NCFChoiceModel,
    "DotProductResponseModel": DotProductResponseModel,
    "TopKSlateGenerator": TopKSlateGenerator,
    "RandomSlateGenerator": RandomSlateGenerator,
    "TopKGreedySlateGenerator": TopKGreedySlateGenerator,
    "CosineSimilarityChoiceModel": CosineSimilarityChoiceModel,
    # "DiverseSlateGenerator": DiverseSlateGenerator,
    # "GreedySlateGenerator": GreedySlateGenerator,
    # "OptimalSlateGenerator": OptimalSlateGenerator,
    "WeightedDotProductResponseModel": WeightedDotProductResponseModel,
    "WeightedCosineResponseModel": WeightedCosineResponseModel,
}
load_dotenv()
