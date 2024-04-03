from scripts.simulation_imports import *


def optimize_model(batch, batch_size):
    (
        state_batch,  # [batch_size, num_item_features]
        selected_doc_feat_batch,  # [batch_size, num_item_features]
        candidates_batch,  # [batch_size, num_candidates, num_item_features]
        satisfaction_batch,  # [batch_size, 1]
        next_state_batch,  # [batch_size, num_item_features]
    ) = batch

    optimizer.zero_grad()

    # Q(s, a): [batch_size, 1]
    q_val = agent.compute_q_values(
        state_batch, selected_doc_feat_batch, use_policy_net=True
    )  # type: ignore

    cand_qtgt_list = []
    for b in range(next_state_batch.shape[0]):
        next_state = next_state_batch[b, :]
        candidates = candidates_batch[b, :, :]
        next_state_rep = next_state.repeat((candidates.shape[0], 1))

        cand_qtgt = agent.compute_q_values(
            next_state_rep, candidates, use_policy_net=False
        )  # type: ignore

        choice_model.score_documents(next_state_rep, candidates)
        scores_tens = (
            torch.Tensor(choice_model.scores).to(DEVICE).unsqueeze(dim=1)
        )  # [num_candidates, 1]
        # retrieve max_a Q(s', a)
        # scores_tens = torch.softmax(scores_tens, dim=0)

        topk = torch.topk((cand_qtgt * scores_tens), dim=0, k=SLATE_SIZE)

        curr_q_tgt = topk.values

        topk_idx = topk.indices
        p_sum = scores_tens[topk_idx, :].squeeze().sum()

        # normalize curr_q_tgt to sum to 1
        curr_q_tgt = torch.sum(curr_q_tgt / p_sum)
        cand_qtgt_list.append(curr_q_tgt)

    q_tgt = torch.stack(cand_qtgt_list).unsqueeze(dim=1)
    expected_q_values = q_tgt * GAMMA + satisfaction_batch.unsqueeze(dim=1)

    loss = criterion(q_val, expected_q_values)

    # Optimize the model
    loss.backward()
    optimizer.step()
    item = actor.compute_proto_slate(state_batch, use_actor_policy_net=True)
    proto_action_tensor = item.reshape(batch_size, SLATE_SIZE, NUM_ITEM_FEATURES)

    actor_loss_list = []

    # Shuffle the tensor along the first dimension (dimension 0)
    # shuffled_tensor = proto_action_tensor[torch.randperm(proto_action_tensor.size(0))]
    # F.cosine_similarity(proto_action_tensor)

    # 'shuffled_tensor' now contains the original tensor with its first dimension shuffled

    # for i in range(batch_size):
    #     item_set = proto_action_tensor[i]
    #     selected_indices = random.sample(range(5), 3)

    #     # Calculate the loss based on the orthogonality of the selected tensors
    #     loss = 0.0
    #     for j in selected_indices:
    #         for k in selected_indices:
    #             if j < k:  # Ensure unique pairs
    #                 # Compute the cosine similarity between the two selected tensors
    #                 similarity = F.cosine_similarity(item_set[j], item_set[k], dim=0)

    #                 # Orthogonality loss term (minimize cosine similarity)
    #                 loss += torch.sum(1 - similarity)
    #     actor_loss_list.append(loss)

    # Taking the average along the third axis to reduce the tensor size
    proto_action_tensor_2 = torch.mean(proto_action_tensor, axis=1)
    for i in range(batch_size):
        proto_action_tensor_rep = proto_action_tensor[i]
        state = state_batch[i]
        user_state_rep = state.repeat((proto_action_tensor_rep.shape[0], 1))
        q_loss = agent.compute_q_values(
            user_state_rep,
            proto_action_tensor_rep,
            use_policy_net=True,
        )
        with torch.no_grad():
            detached_scores = choice_model._score_documents(
                user_state_rep, proto_action_tensor_rep
            ).detach()
            # [num_candidates, 1]
        scores_tens_loss = torch.Tensor(detached_scores).to(DEVICE).unsqueeze(dim=1)
        # max over Q(s', a)
        scores_tens_loss = torch.softmax(scores_tens_loss, dim=0)
        v_sum = scores_tens_loss.squeeze().sum()
        actor_loss_list.append(-torch.sum((q_loss * scores_tens_loss) / v_sum))
    actor_loss = torch.tensor(actor_loss_list, requires_grad=True).unsqueeze(1).mean()
    # actor_item_loss = torch.empty(128, 5)
    # for i in range(5):
    #     q_values = -agent.compute_q_values(
    #         state_batch,
    #         proto_action_tensor[:, i, :],
    #         use_policy_net=True,
    # print(q_loss.grad_fn)
    # print(scores_tens_loss.grad_fn)
    # #     )
    # a = 10
    #     if q_values.shape != (128, 1):
    #         raise ValueError(f"Shape mismatch in q_values for column {i}")

    #     # Assign q_values to the corresponding column
    #     actor_item_loss[:, i : i + 1] = -q_values
    # actor_loss = torch.mean(actor_item_loss, dim=1, keepdim=True).mean()
    actor_loss.backward()
    actor_optimizer.step()
    return loss, actor_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config_path = "src/scripts/config.yaml"
    parser.add_argument(
        "--config",
        type=str,
        default=config_path,
        help="Path to the config file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    parameters = config["parameters"]
    SEEDS = parameters["seeds"]
    for seed in SEEDS:
        pl.seed_everything(seed)
        resp_amp_factor = parameters["resp_amp_factor"]

        ######## Training related parameters ########
        NUM_CANDIDATES = parameters["num_candidates"]
        NUM_ITEM_FEATURES = parameters["num_item_features"]
        NEAREST_NEIGHBOURS = parameters["nearest_neighbours"]
        # NUM_ITEM_FEATURES = parameters["num_item_features"]
        SLATE_SIZE = parameters["slate_size"]
        REPLAY_MEMORY_CAPACITY = parameters["replay_memory_capacity"]
        BATCH_SIZE = parameters["batch_size"]
        GAMMA = parameters["gamma"]
        TAU = parameters["tau"]
        LR = float(parameters["lr"])
        NUM_EPISODES = parameters["num_episodes"]
        WARMUP_BATCHES = parameters["warmup_batches"]
        DEVICE = parameters["device"]
        ALPHA_RESPONSE = parameters["alpha_response"]
        DEVICE = torch.device(DEVICE)
        print("DEVICE: ", DEVICE)
        ######## Models related parameters ########
        slate_gen_model_cls = parameters["slate_gen_model_cls"]
        choice_model_cls = parameters["choice_model_cls"]
        response_model_cls = parameters["response_model_cls"]

        ######## Init_wandb ########
        RUN_NAME = (
            f"Mind_Dataset_GAMMA_{GAMMA}_SEED_{seed}_ALPHA_{ALPHA_RESPONSE}_SLATE_WP"
        )
        wandb.init(project="mind_dataset", config=config["parameters"], name=RUN_NAME)

        ################################################################
        user_state = UserState(device=DEVICE)
        slate_gen_model_cls = class_name_to_class[slate_gen_model_cls]
        choice_model_cls = class_name_to_class[choice_model_cls]
        response_model_cls = class_name_to_class[response_model_cls]
        slate_gen = slate_gen_model_cls(slate_size=SLATE_SIZE)

        choice_model_kwgs = {"device": DEVICE}
        response_model_kwgs = {
            "amp_factor": resp_amp_factor,
            "alpha": ALPHA_RESPONSE,
            "device": DEVICE,
        }
        # input features are 2 * NUM_ITEM_FEATURES since we concatenate the state and one item
        agent = DQNAgent(
            slate_gen=slate_gen,
            input_size=2 * NUM_ITEM_FEATURES,
            output_size=1,
            tau=TAU,
        ).to(DEVICE)

        transition_cls = Transition
        replay_memory_dataset = ReplayMemoryDataset(
            capacity=REPLAY_MEMORY_CAPACITY, transition_cls=transition_cls
        )
        replay_memory_dataloader = DataLoader(
            replay_memory_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=replay_memory_dataset.collate_fn,
            shuffle=False,
        )
        actor = ActorAgentSlate(
            nn_dim=[
                50,
                100,
                150,
                200,
                250,
                300,
                350,
                400,
                450,
                500,
            ],  # observable= [20, 40, 60, 80, 100], weight_decay=1e-4
            k=int(NEAREST_NEIGHBOURS / SLATE_SIZE),
            slate_size=SLATE_SIZE,
        ).to(DEVICE)

        criterion = torch.nn.SmoothL1Loss()
        optimizer = optim.Adam(agent.parameters(), lr=LR)
        actor_optimizer = optim.Adam(actor.parameters(), lr=LR)
        choice_model = choice_model_cls()
        response_model = response_model_cls(**response_model_kwgs)
        env = SlateGym(
            user_state=user_state,
            choice_model=choice_model,
            response_model=response_model,
            device=DEVICE,
        )

        ############################## TRAINING ###################################
        save_dict = defaultdict(list)
        is_terminal = False
        for i_episode in tqdm(range(NUM_EPISODES)):
            satisfaction, loss, diff_to_best, quality, actor_loss = (
                [],
                [],
                [],
                [],
                [],
            )

            env.reset()
            env.hidden_state()
            is_terminal = False
            cum_satisfaction = 0

            candidate_docs = env.get_candidate_docs().to(DEVICE)
            clicked_docs = env.get_clicked_docs().to(DEVICE)

            user_observed_state = env.curr_user.to(DEVICE)
            env.diversity()
            # user_state = user_state / user_state.sum()

            max_sess, avg_sess = [], []
            for i in range(len(clicked_docs)):
                with torch.no_grad():
                    ########################################
                    # satisfactions_candidates = (
                    #     (1 - ALPHA_RESPONSE)
                    #     * torch.mm(
                    #         user_state.unsqueeze(0),
                    #         cdocs_features.t(),
                    #     )
                    #     + ALPHA_RESPONSE * cdocs_quality
                    # ).squeeze(0) * resp_amp_factor

                    # max_rew = satisfactions_candidates.max()
                    # min_rew = satisfactions_candidates.min()
                    # mean_rew = satisfactions_candidates.mean()
                    # std_rew = satisfactions_candidates.std()

                    # max_sess.append(max_rew)
                    # avg_sess.append(mean_rew)
                    ########################################
                    a = 0

                    cdocs_features_act, candidates = actor.k_nearest(
                        user_observed_state,
                        candidate_docs,
                        slate_size=SLATE_SIZE,
                        use_actor_policy_net=True,
                    )
                    user_state_rep = user_observed_state.repeat(
                        (cdocs_features_act.shape[0], 1)
                    ).to(DEVICE)

                    q_val = agent.compute_q_values(
                        state=user_state_rep,
                        candidate_docs_repr=cdocs_features_act,
                        use_policy_net=True,
                    )  # type: ignore

                    choice_model.score_documents(
                        user_state=user_state_rep, docs_repr=cdocs_features_act
                    )
                    scores = torch.Tensor(choice_model.scores).to(DEVICE)
                    # scores = torch.softmax(scores, dim=0)

                    q_val = q_val.squeeze()
                    slate = agent.get_action(scores, q_val)
                    # print("slate: ", slate)

                    (
                        selected_doc_feature,
                        response,
                        is_terminal,
                        next_user_state,
                        _,
                        _,
                        diverse_score,
                        user_satisfaction,
                        relevance,
                    ) = env.step(
                        slate, iterator=i, cdocs_subset_idx=candidates.to(DEVICE)
                    )
                    # normalize satisfaction between 0 and 1
                    # response = (response - min_rew) / (max_rew - min_rew)
                    satisfaction.append(response)

                    # check that not null document has been selected
                    if not torch.all(selected_doc_feature == 0):
                        # append 4 document length

                        # push memory
                        replay_memory_dataset.push(
                            transition_cls(
                                user_observed_state,  # type: ignore
                                selected_doc_feature,
                                cdocs_features_act,
                                response,
                                next_user_state,  # type: ignore
                            )
                        )

                    user_observed_state = next_user_state
                    # user_state = user_state / user_state.sum()

            # optimize model
            if len(replay_memory_dataset.memory) >= WARMUP_BATCHES * BATCH_SIZE:
                batch = next(iter(replay_memory_dataloader))
                for elem in batch:
                    elem.to(DEVICE)
                batch_loss, actor_batch_loss = optimize_model(batch, BATCH_SIZE)
                agent.soft_update_target_network()
                loss.append(batch_loss)
                actor_loss.append(actor_batch_loss)

            loss = torch.mean(torch.tensor(loss))
            actor_loss = torch.mean(torch.tensor(actor_loss))
            # sess_length = np.sum(time_unit_consumed)
            ep_quality = torch.mean(torch.tensor(quality))
            ep_avg_satisfaction = torch.mean(torch.tensor(satisfaction))
            ep_cum_satisfaction = torch.sum(torch.tensor(satisfaction))
            # ep_max_avg = torch.mean(torch.tensor(max_sess))
            # ep_max_cum = torch.sum(torch.tensor(max_sess))
            # ep_avg_avg = torch.mean(torch.tensor(avg_sess))
            # ep_avg_cum = torch.sum(torch.tensor(avg_sess))
            # cum_normalized = (
            #     ep_cum_satisfaction / ep_max_cum
            #     if ep_max_cum > 0
            #     else ep_max_cum / ep_cum_satisfaction
            # )

            log_str = (
                f"Loss: {loss}\n"
                f"Avg_satisfaction: {ep_avg_satisfaction} - Cum_Rew: {ep_cum_satisfaction}\n"
                #     f"Max_Avg_satisfaction: {ep_max_avg} - Max_Cum_Rew: {ep_max_cum}\n"
                #     f"Avg_Avg_satisfaction: {ep_avg_avg} - Avg_Cum_Rew: {ep_avg_cum}\n"
                #     f"Cumulative_Normalized: {cum_normalized}"
                #
                f"Diverse_score: {diverse_score}\n"
            )
            print(log_str)
            ###########################################################################
            log_dict = {
                "loss": loss,
                "actor_loss": actor_loss,
                # "quality": ep_quality,
                "quality": ep_quality,
                "avg_satisfaction": ep_avg_satisfaction,
                "cum_satisfaction": ep_cum_satisfaction,
                # "max_avg": ep_max_avg,
                # "max_cum": ep_max_cum,
                # "avg_avg": ep_avg_avg,
                # "avg_cum": ep_avg_cum,
                # "best_rl_avg_diff": ep_max_avg - ep_avg_satisfaction,
                # "best_avg_avg_diff": ep_max_avg - ep_avg_avg,
                # "cum_normalized": cum_normalized,
                "diverse_score": diverse_score,
            }
            if len(replay_memory_dataset.memory) >= (WARMUP_BATCHES * BATCH_SIZE):
                log_dict["loss"] = loss
            wandb.log(log_dict, step=i_episode)

            # ###########################################################################
            # save_dict["session_length"].append(sess_length)
            # save_dict["ep_cum_satisfaction"].append(ep_cum_satisfaction)
            # save_dict["ep_avg_satisfaction"].append(ep_avg_satisfaction)
            # save_dict["loss"].append(loss)
            # save_dict["best_rl_avg_diff"].append(ep_max_avg - ep_avg_satisfaction)
            # save_dict["best_avg_avg_diff"].append(ep_max_avg - ep_avg_avg)
            # save_dict["cum_normalized"].append(cum_normalized)

        wandb.finish()
        directory = f"div_entropy_wpslate_{ALPHA_RESPONSE}_gamma"
        save_run_wa(
            seed=seed,
            save_dict=save_dict,
            agent=agent,
            actor=actor,
            directory=directory,
        )
