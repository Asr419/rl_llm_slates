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


import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset

from rl_mind_dataset.utils import save_run_ncf
from tqdm import tqdm


class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=64):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_ids, item_ids, train=True):
        # print(user_ids)
        user_embeds = user_ids  # torch.tensor([embedding_dict_1[item_id] for item_id in user_ids])
        item_embeds = item_ids  # torch.tensor([embedding_dict_1[item_id] for item_id in item_ids])  # Retrieve item embeddings
        if train:
            concat_embeds = torch.cat([user_embeds, item_embeds], dim=1)
        else:
            concat_embeds = torch.cat([user_embeds, item_embeds], dim=0)
        output = self.fc_layers(concat_embeds)
        return output.squeeze()


class DataFrameDataset(Dataset):
    def __init__(self, dataframe, embedding_dict):
        self.data = dataframe
        self.dict = embedding_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # print(row['user_embedding'])
        default_embedding = np.zeros(50, dtype=np.float32)
        user_id = torch.tensor(self.dict.get(row["user_embedding"], default_embedding))
        item_id = torch.tensor(self.dict.get(row["click_history"], default_embedding))
        rating = torch.tensor(row["click"]).to(torch.float32)
        return {"user_id": user_id, "item_id": item_id, "rating": rating}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_PATH = Path.home() / Path(os.environ.get("DATA_PATH"))
    # device = torch.device("cpu")
    interactions_path_data = DATA_PATH / Path(
        "MINDlarge_train/choice_model_data.feather"
    )
    interactions2 = pd.read_feather(interactions_path_data)
    dataset_reader = DatasetReader()
    if torch.cuda.is_available():
        # Get the current CUDA device index
        device_idx = torch.cuda.current_device()
        # Get the name of the CUDA device
        device_name = torch.cuda.get_device_name(device_idx)
        print(f"Using CUDA device: {device_name}")
    else:
        print("CUDA is not available. Using CPU.")

    embedding_dict, all_item_vectors = dataset_reader.item2vecdict()
    dataset = DataFrameDataset(interactions2, embedding_dict)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    num_users = len(set(interactions2["user_embedding"]))
    num_items = len(set(interactions2["click_history"]))

    # Instantiate the NCF model
    model = NCF(
        num_users=num_users, num_items=num_items, embedding_dim=50, hidden_dim=50
    )

    # Define loss function and optimizer
    criterion = (
        nn.BCEWithLogitsLoss()
    )  # Binary cross-entropy loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for batch in train_loader:

            user_ids = batch["user_id"]

            item_ids = batch["item_id"]
            ratings = batch["rating"]

            optimizer.zero_grad()
            outputs = model(user_ids, item_ids)
            # print(outputs.dtype)
            # print(ratings.dtype)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
        print(f"epoch: {epoch}, loss: {loss}")
    directory = f"user_choice_model"
    save_run_ncf(agent=model, directory=directory)
