import time

import pandas as pd

from scripts.simulation_imports import *

save_path = os.environ.get("SAVE_PATH")
BASE_LOAD_PATH = Path.home() / save_path
load_dotenv()
base_path = Path.home() / Path(os.environ.get("SAVE_PATH"))
MODEL_SEED = 5

DEVICE = "cpu"
print("DEVICE: ", DEVICE)

NUM_CANDIDATES = [300]
if __name__ == "__main__":
    seed = 37
    ALPHA = 0.0

    model_name_list = []
    num_candidates_list = []
    serving_time_users_list = []

    FOLDER_NAME = f"div_entropy_wpslate_{ALPHA}_gamma_{MODEL_SEED}"
    AGENT_PATH = base_path / FOLDER_NAME / Path("model.pt")
    ACTOR_PATH = base_path / FOLDER_NAME / Path("actor.pt")
    parser = argparse.ArgumentParser()
    config_path = base_path / FOLDER_NAME / Path("config.yaml")
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

    for num_candidates in NUM_CANDIDATES:
        num_candidates_list.append(num_candidates)
        model_name_list.append(f"WA-SlateQ")

        ######## User related parameters ########

        choice_model_cls = parameters["choice_model_cls"]
        response_model_cls = parameters["response_model_cls"]
        resp_amp_factor = parameters["resp_amp_factor"]

        ######## Environment related parameters ########
        SLATE_SIZE = parameters["slate_size"]
        NUM_USERS = 5
        NUM_ITEM_FEATURES = parameters["num_item_features"]
        SESS_BUDGET = parameters["sess_budget"]
        NUM_USER_FEATURES = parameters["num_user_features"]
        ALPHA_RESPONSE = parameters["alpha_response"]

        ######## Training related parameters ########
        REPLAY_MEMORY_CAPACITY = parameters["replay_memory_capacity"]
        BATCH_SIZE = parameters["batch_size"]
        GAMMA = parameters["gamma"]
        TAU = parameters["tau"]
        LR = float(parameters["lr"])
        NUM_EPISODES = 1
        WARMUP_BATCHES = parameters["warmup_batches"]
        DEVICE = parameters["device"]
        DEVICE = torch.device(DEVICE)
        print("DEVICE: ", DEVICE)
        ######## Models related parameters ########
        slate_gen_model_cls = parameters["slate_gen_model_cls"]

        ################################################################

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
        agent = torch.load(AGENT_PATH, map_location="cpu").to(DEVICE)

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
        user_serving_time = []
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
            serving_time = []
            for i in range(len(clicked_docs)):
                start = time.time()
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

                    # q_val = agent.compute_q_values(
                    #     state=user_state_rep,
                    #     candidate_docs_repr=cdocs_features_act,
                    #     use_policy_net=True,
                    # )  # type: ignore
                    q_val_list = []
                    for cdoc in cdocs_features_act:
                        q_val = agent.compute_q_values(
                            state=user_observed_state.unsqueeze(dim=0),
                            candidate_docs_repr=cdoc.unsqueeze(dim=0),
                            use_policy_net=True,
                        )  # type: ignore
                        q_val_list.append(q_val)
                    q_val = torch.stack(q_val_list).to(DEVICE)

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
                    end = time.time()
                    serving_time.append(end - start)
                    # normalize satisfaction between 0 and 1
                    # response = (response - min_rew) / (max_rew - min_rew)
                    quality.append(0.0)

                    # for row1 in candidate_docs[slate, :]:
                    #     for row2 in actual_selected_items:
                    #         if torch.all(torch.eq(row1, row2)):
                    #             selected_doc_feature = row1
                    #             response, user_satisfaction, relevance = (
                    #                 response_model._generate_response(
                    #                     user_state._generate_hidden_state().to(DEVICE),
                    #                     selected_doc_feature.to(DEVICE),
                    #                     row2,
                    #                     diversity=diverse_score,
                    #                     alpha=alpha,
                    #                 )
                    #             )
                    #             clicked_docs_lists = [
                    #                 tensor.tolist() for tensor in clicked_docs
                    #             ]
                    #             index = clicked_docs_lists.index(row2.tolist())
                    #             actual_selected_items = torch.cat(
                    #                 (clicked_docs[:index], clicked_docs[index + 1 :]),
                    #                 dim=0,
                    #             )

                    #             quality.pop()
                    #             quality.append(1.0)
                    #             break

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
                    # push memory
                    # replay_memory_dataset.push(
                    #     transition_cls(
                    #         user_state,  # type: ignore
                    #         selected_doc_feature,
                    #         cdocs_features,
                    #         response,
                    #         next_user_state,  # type: ignore
                    #     )
                    # )

            user_serving_time.append(np.mean(serving_time))

        print("num_candidates: ", num_candidates)
        print("mean serving time: ", np.mean(user_serving_time))
        serving_time_users_list.append(np.mean(user_serving_time))
    # construct a df and save it
    res_df = pd.DataFrame(
        zip(model_name_list, num_candidates_list, serving_time_users_list),
        columns=["model_name", "num_candidates", "serving_time"],
    )
    res_df.to_csv(BASE_LOAD_PATH / "serving_time_wp_slate.csv", index=False)
    print(res_df)
