import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()


def save_run(seed, agent, save_dict, directory: str):
    save_path = Path(os.environ.get("LLM_TRAINED_PATH"))  # type: ignore
    save_path = Path.home() / save_path
    save_path.mkdir(parents=True, exist_ok=True)

    time_now = datetime.now().strftime("%m-%d_%H-%M-%S")
    directory = directory + "_" + str(seed)

    # Create the directory with the folder name
    path = Path(directory)
    save_dir = Path(save_path / path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # save config
    source_path = "scripts/config.yaml"
    destination_path = save_dir / Path("config.yaml")
    shutil.copy(source_path, destination_path)

    # Save the model
    model_save_name = f"model.pt"
    torch.save(agent, save_dir / Path(model_save_name))

    # save logs dict
    logs_save_name = Path(f"logs_dict.pickle")
    with open(save_dir / logs_save_name, "wb") as f:
        pickle.dump(save_dict, f)
    print(f"Run saved successfully in: {save_dir}")


def save_run_wa(seed, agent, save_dict, actor, directory: str):
    save_path = Path(os.environ.get("LLM_TRAINED_PATH"))  # type: ignore
    save_path = Path.home() / save_path
    save_path.mkdir(parents=True, exist_ok=True)

    time_now = datetime.now().strftime("%m-%d_%H-%M-%S")
    directory = directory + "_" + str(seed)

    # Create the directory with the folder name
    path = Path(directory)
    save_dir = Path(save_path / path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # save config
    source_path = "scripts/config.yaml"
    destination_path = save_dir / Path("config.yaml")
    shutil.copy(source_path, destination_path)

    # Save the model
    model_save_name = f"model.pt"
    torch.save(agent, save_dir / Path(model_save_name))
    actor_save_name = f"actor.pt"
    torch.save(actor, save_dir / Path(actor_save_name))

    # save logs dict
    logs_save_name = Path(f"logs_dict.pickle")
    with open(save_dir / logs_save_name, "wb") as f:
        pickle.dump(save_dict, f)
    print(f"Run saved successfully in: {save_dir}")


def save_run_ncf(agent, directory: str):
    save_path = Path(os.environ.get("SAVE_PATH"))  # type: ignore
    save_path = Path.home() / save_path
    save_path.mkdir(parents=True, exist_ok=True)

    time_now = datetime.now().strftime("%m-%d_%H-%M-%S")
    # directory = directory + "_" + str(seed)

    # Create the directory with the folder name
    path = Path(directory)
    save_dir = Path(save_path / path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the model
    model_save_name = f"model.pt"
    torch.save(agent, save_dir / Path(model_save_name))

    print(f"Run saved successfully in: {save_dir}")


def test_save_run(seed, save_dict, directory: str):
    save_path = Path(os.environ.get("TEST_PATH"))  # type: ignore
    save_path = Path.home() / save_path
    save_path.mkdir(parents=True, exist_ok=True)

    time_now = datetime.now().strftime("%m-%d_%H-%M-%S")
    directory = directory + "_" + str(seed)

    # Create the directory with the folder name
    path = Path(directory)
    save_dir = Path(save_path / path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # # save config
    # source_path = "src/scripts/config.yaml"
    # destination_path = save_dir / Path("config.yaml")
    # shutil.copy(source_path, destination_path)

    # # Save the model
    # model_save_name = f"model.pt"
    # torch.save(agent, save_dir / Path(model_save_name))

    # save logs dict
    logs_save_name = Path(f"logs_dict.pickle")
    with open(save_dir / logs_save_name, "wb") as f:
        pickle.dump(save_dict, f)
    print(f"Run saved successfully in: {save_dir}")
