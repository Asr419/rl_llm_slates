import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
import os
from collections import defaultdict
import pytorch_lightning as pl
from scripts.simulation_imports import *
from rl_mind_dataset.user_modelling.ncf import NCF, DataFrameDataset
from openai import OpenAI


def optimize_model(
    batch, agent, choice_model, optimizer, criterion, GAMMA, SLATE_SIZE, DEVICE
):
    (
        state_batch,
        selected_doc_feat_batch,
        candidates_batch,
        satisfaction_batch,
        next_state_batch,
    ) = batch

    optimizer.zero_grad()

    # Q(s, a)
    q_val = agent.compute_q_values(
        state_batch, selected_doc_feat_batch, use_policy_net=True
    )

    cand_qtgt_list = []
    for b in range(next_state_batch.shape[0]):
        next_state = next_state_batch[b]
        candidates = candidates_batch[b]
        next_state_rep = next_state.unsqueeze(0).expand_as(candidates)

        cand_qtgt = agent.compute_q_values(
            next_state_rep, candidates, use_policy_net=False
        )
        choice_model.score_documents(next_state_rep, candidates)

        scores_tens = torch.tensor(choice_model.scores, device=DEVICE).unsqueeze(1)
        topk = torch.topk((cand_qtgt * scores_tens), k=SLATE_SIZE, dim=0)

        p_sum = scores_tens[topk.indices].squeeze().sum()
        curr_q_tgt = torch.sum(topk.values / p_sum)
        cand_qtgt_list.append(curr_q_tgt)

    q_tgt = torch.stack(cand_qtgt_list).unsqueeze(1)
    expected_q_values = q_tgt * GAMMA + satisfaction_batch.unsqueeze(1)
    loss = criterion(q_val, expected_q_values)

    loss.backward()
    optimizer.step()
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="scripts/config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    parameters = config["parameters"]
    SEEDS = parameters["seeds"]

    # Extract all parameters once
    NUM_CANDIDATES = parameters["num_candidates"]
    NUM_ITEM_FEATURES = parameters["num_item_features"]
    SLATE_SIZE = parameters["slate_size"]
    REPLAY_MEMORY_CAPACITY = parameters["replay_memory_capacity"]
    BATCH_SIZE = parameters["batch_size"]
    GAMMA = parameters["gamma"]
    TAU = parameters["tau"]
    LR = float(parameters["lr"])
    NUM_EPISODES = parameters["num_episodes"]
    WARMUP_BATCHES = parameters["warmup_batches"]
    DEVICE = torch.device(parameters["device"])
    ALPHA_RESPONSE = parameters["alpha_response"]
    resp_amp_factor = parameters["resp_amp_factor"]
    print(DEVICE)

    for seed in SEEDS:
        pl.seed_everything(seed)

        wandb.init(
            project="recsys_llm_reasoning",
            config=parameters,
            name=f"GAMMA_{GAMMA}_SEED_{seed}_ALPHA_{ALPHA_RESPONSE}_SLATEQ",
        )

        # Initialize models
        user_state = UserState(device=DEVICE)
        slate_gen = class_name_to_class[parameters["slate_gen_model_cls"]](
            slate_size=SLATE_SIZE
        )

        agent = DQNAgent(
            slate_gen=slate_gen,
            input_size=2 * NUM_ITEM_FEATURES,
            output_size=1,
            tau=TAU,
        ).to(DEVICE)

        choice_model = class_name_to_class[parameters["choice_model_cls"]](
            device=DEVICE
        )
        response_model = class_name_to_class[parameters["response_model_cls"]](
            amp_factor=resp_amp_factor, alpha=ALPHA_RESPONSE, device=DEVICE
        )

        # Initialize memory and optimizer
        replay_memory_dataset = ReplayMemoryDataset(
            capacity=REPLAY_MEMORY_CAPACITY, transition_cls=Transition
        )

        replay_memory_dataloader = DataLoader(
            replay_memory_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=replay_memory_dataset.collate_fn,
            shuffle=False,
            pin_memory=True,  # Faster data transfer to GPU
        )

        optimizer = optim.Adam(agent.parameters(), lr=LR)
        criterion = torch.nn.SmoothL1Loss()

        # OpenAI client
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY_4"),
            base_url=os.getenv("DEEPSEEK_BASE_URL_R"),
        )

        # Training loop
        for i_episode in tqdm(range(NUM_EPISODES)):
            env = SlateGym(
                user_state=user_state,
                choice_model=choice_model,
                response_model=response_model,
                device=DEVICE,
            )

            env.reset()
            env.hidden_state()

            candidate_docs = env.get_candidate_docs().to(DEVICE)
            clicked_docs = env.get_clicked_docs().to(DEVICE)
            user_observed_state = env.curr_user.to(DEVICE)
            env.diversity()

            satisfaction = []
            for i in range(len(clicked_docs)):
                with torch.no_grad():
                    user_state_rep = user_observed_state.unsqueeze(0).expand(
                        candidate_docs.size(0), -1
                    )

                    q_val = agent.compute_q_values(
                        state=user_state_rep,
                        candidate_docs_repr=candidate_docs,
                        use_policy_net=True,
                    ).squeeze()

                    choice_model.score_documents(user_state_rep, candidate_docs)
                    scores = choice_model.scores.clone().detach().to(DEVICE)
                    # scores = torch.tensor(choice_model.scores, device=DEVICE)

                    slate = agent.get_action(scores, q_val)

                    (
                        selected_doc_feature,
                        response,
                        is_terminal,
                        next_user_state,
                        *_,
                    ) = env.step(
                        slate, iterator=i, cdocs_subset_idx=None, client=client
                    )

                    satisfaction.append(response)

                    if not torch.all(selected_doc_feature == 0):
                        replay_memory_dataset.push(
                            Transition(
                                user_observed_state,
                                selected_doc_feature,
                                candidate_docs,
                                response,
                                next_user_state,
                            )
                        )

                    user_observed_state = next_user_state

            # Optimization
            if len(replay_memory_dataset.memory) >= WARMUP_BATCHES * BATCH_SIZE:
                batch = next(iter(replay_memory_dataloader))
                batch = [t.to(DEVICE) for t in batch]
                loss = optimize_model(
                    batch,
                    agent,
                    choice_model,
                    optimizer,
                    criterion,
                    GAMMA,
                    SLATE_SIZE,
                    DEVICE,
                )
                agent.soft_update_target_network()
            else:
                loss = torch.tensor(0.0)

            # Logging
            ep_avg_satisfaction = torch.mean(torch.tensor(satisfaction))
            ep_cum_satisfaction = torch.sum(torch.tensor(satisfaction))

            wandb.log(
                {
                    "avg_satisfaction": ep_avg_satisfaction,
                    "cum_satisfaction": ep_cum_satisfaction,
                    "loss": loss,
                    "diverse_score": env.diversity_value,
                },
                step=i_episode,
            )

        wandb.finish()
        save_run(seed, {}, agent, f"slateq_{ALPHA_RESPONSE}")
