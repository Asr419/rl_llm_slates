from scripts.simulation_imports import *
from rl_mind_dataset.user_modelling.ncf import NCF, DataFrameDataset

DEVICE = "cpu"
print("DEVICE: ", DEVICE)
load_dotenv()
base_path = Path.home() / Path(os.environ.get("SAVE_PATH"))
if __name__ == "__main__":
    USER_SEED = 11
    SEEDS = [33, 99]
    NUM_EPISODES = 1000
    for seed in tqdm(SEEDS):

        ALPHA = 0.0
        RUN_BASE_PATH = Path(f"div_entropy_wpslate_{ALPHA}_gamma_{seed}")
        parser = argparse.ArgumentParser()
        config_path = base_path / RUN_BASE_PATH / Path("config.yaml")
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
        pl.seed_everything(USER_SEED)
        PATH = base_path / RUN_BASE_PATH / Path("model.pt")
        ACTOR_PATH = base_path / RUN_BASE_PATH / Path("actor.pt")
        resp_amp_factor = parameters["resp_amp_factor"]

        ######## Training related parameters ########
        NUM_CANDIDATES = parameters["num_candidates"]
        NUM_ITEM_FEATURES = parameters["num_item_features"]
        # NUM_ITEM_FEATURES = parameters["num_item_features"]
        SLATE_SIZE = parameters["slate_size"]
        REPLAY_MEMORY_CAPACITY = parameters["replay_memory_capacity"]
        BATCH_SIZE = parameters["batch_size"]
        GAMMA = parameters["gamma"]
        TAU = parameters["tau"]
        LR = float(parameters["lr"])

        WARMUP_BATCHES = parameters["warmup_batches"]

        ALPHA_RESPONSE = parameters["alpha_response"]

        ######## Models related parameters ########
        slate_gen_model_cls = parameters["slate_gen_model_cls"]
        choice_model_cls = parameters["choice_model_cls"]
        response_model_cls = parameters["response_model_cls"]

        # RUN_NAME = f"ColdStartTest_WP_SLATE"
        # wandb.init(project="mind_dataset", config=config["parameters"], name=RUN_NAME)

        user_state = UserState(
            device=DEVICE,
            test=True,
            specialist=True,
            generalist=False,
            cold_start=False,
        )
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
        agent = torch.load(PATH, map_location="cpu").to(DEVICE)

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
        actor = torch.load(ACTOR_PATH, map_location="cpu").to(DEVICE)

        criterion = torch.nn.SmoothL1Loss()
        optimizer = optim.Adam(agent.parameters(), lr=LR)
        choice_model = choice_model_cls(**choice_model_kwgs)
        response_model = response_model_cls(**response_model_kwgs)
        env = SlateGym(
            user_state=user_state,
            choice_model=choice_model,
            response_model=response_model,
            device=DEVICE,
        )
        # torch.cuda.set_device(DEVICE)

        ############################## TRAINING ###################################
        save_dict = defaultdict(list)
        is_terminal = False
        for i_episode in tqdm(range(NUM_EPISODES)):
            satisfaction, loss, diff_to_best, quality, time_unit_consumed = (
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
            actual_selected_items = clicked_docs

            user_observed_state = env.curr_user.to(DEVICE)
            alpha = env.diversity()

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
                    quality.append(0.0)

                    for row1 in candidate_docs[slate, :]:
                        for row2 in actual_selected_items:
                            if torch.all(torch.eq(row1, row2)):
                                selected_doc_feature = row1
                                response, user_satisfaction, relevance = (
                                    response_model._generate_response(
                                        user_state._generate_hidden_state().to(DEVICE),
                                        selected_doc_feature.to(DEVICE),
                                        row2,
                                        diversity=diverse_score,
                                        alpha=alpha,
                                    )
                                )
                                clicked_docs_lists = [
                                    tensor.tolist() for tensor in clicked_docs
                                ]
                                index = clicked_docs_lists.index(row2.tolist())
                                actual_selected_items = torch.cat(
                                    (clicked_docs[:index], clicked_docs[index + 1 :]),
                                    dim=0,
                                )

                                quality.pop()
                                quality.append(1.0)
                                break

                    next_user_state = user_state.update_state(
                        selected_doc_feature=selected_doc_feature.to(DEVICE)
                    )
                    satisfaction.append(response)
                    # check that not null document has been selected
                    # if not torch.all(selected_doc_feature == 0):
                    #     # append 4 document length

                    #     # push memory
                    #     replay_memory_dataset.push(
                    #         transition_cls(
                    #             user_observed_state,  # type: ignore
                    #             selected_doc_feature,
                    #             candidate_docs,
                    #             response,
                    #             next_user_state,  # type: ignore
                    #         )
                    #     )

                    user_observed_state = next_user_state
                    # user_state = user_state / user_state.sum()

            sess_length = np.sum(time_unit_consumed)
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

            # wandb.log(log_dict, step=i_episode)

            #     # ###########################################################################
            save_dict["hit_documents"].append(ep_quality)
            save_dict["ep_cum_satisfaction"].append(ep_cum_satisfaction)
            save_dict["ep_avg_satisfaction"].append(ep_avg_satisfaction)
            save_dict["diverse_score"].append(diverse_score)
            save_dict["user_satisfaction"].append(user_satisfaction)
            save_dict["relevance"].append(relevance)
            save_dict["entropy_diversity"].append(alpha)
        #     # save_dict["loss"].append(loss)
        #     # save_dict["best_rl_avg_diff"].append(ep_max_avg - ep_avg_satisfaction)
        #     # save_dict["best_avg_avg_diff"].append(ep_max_avg - ep_avg_avg)
        #     # save_dict["cum_normalized"].append(cum_normalized)

        # wandb.finish()
        directory = f"wp_slate_specialist_diversity"
        test_save_run(seed=seed, save_dict=save_dict, directory=directory)
