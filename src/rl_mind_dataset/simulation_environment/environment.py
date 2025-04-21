import random as rand
from math import sqrt
from typing import Optional
import re
import logging
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch


class SlateGym(gym.Env):
    def __init__(
        self,
        user_state,
        choice_model,
        response_model,
        num_candidates: int = 10,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.user_state = user_state
        self.device = device
        self.num_candidates = num_candidates
        self.choice_model = choice_model
        self.response_model = response_model

        # initialized by reset
        self.is_terminal = False
        self.curr_user = torch.Tensor()
        self.hidden_choice_state = torch.Tensor()
        self.candidate_docs = torch.Tensor()
        self.clicked_docs = torch.Tensor()
        self.diversity_value = torch.Tensor()

        self.LOG_DIR = Path("reasoning_logs/logs_wolpertinger")
        self.LOG_DIR.mkdir(exist_ok=True)
        self.row_counter = 0
        self.current_log_file = None  # Track current log file
        self.logger = self.setup_logger()

    def step(
        self,
        slate: torch.Tensor,
        iterator: int,
        cdocs_subset_idx: Optional[torch.Tensor] = None,
        test=False,
        client=None,
    ):
        # Move tensors to device once
        if cdocs_subset_idx is not None:
            cdocs_subset_idx = cdocs_subset_idx.to(self.device)

        self.candidate_docs = self.candidate_docs.to(self.device)
        cdocs_feature = (
            self.candidate_docs[cdocs_subset_idx, :]
            if cdocs_subset_idx is not None
            else self.candidate_docs
        )
        # if cdocs_subset_idx is not None:
        #     self.candidate_docs = self.candidate_docs[cdocs_subset_idx, :]
        cdocs_feature = cdocs_feature[slate, :].to(self.device)

        # Process LLM response
        # cdocs_feature = self.modified_llm_response(cdocs_feature, client=client)

        # Calculate diversity score
        diverse_score = self.user_state.categorical_diversity(cdocs_feature)

        # Prepare hidden state
        hidden_step_state = self.clicked_docs[iterator].to(self.device)
        hidden_step_state_rep = hidden_step_state.expand(cdocs_feature.shape[0], -1)

        # Document selection
        self.choice_model.score_documents(hidden_step_state_rep, cdocs_feature)
        selected_doc_idx = self.choice_model.choose_document()

        if selected_doc_idx == self.choice_model.no_selection_token:
            response = self.response_model.generate_null_response()
            user_satisfaction = torch.tensor(0, device=self.device)
            relevance = torch.tensor(0, device=self.device)
            selected_doc_feature = torch.zeros(
                cdocs_feature.shape[1], device=self.device
            )
        else:
            selected_doc_feature = cdocs_feature[selected_doc_idx, :]
            response, user_satisfaction, relevance = (
                self.response_model._generate_response(
                    self.hidden_choice_state.to(self.device),
                    selected_doc_feature,
                    hidden_step_state,
                    diversity=diverse_score,
                    alpha=self.diversity_value,
                )
            )

        # Update user state
        next_user_state = (
            torch.zeros(50, device=self.device)
            if test
            else self.user_state.update_state(selected_doc_feature)
        )

        # Check terminal condition
        self.is_terminal = len(self.clicked_docs) == iterator

        return (
            selected_doc_feature,
            response,
            self.is_terminal,
            next_user_state,
            False,
            {},
            diverse_score,
            user_satisfaction,
            relevance,
            self.candidate_docs,
            cdocs_feature,
        )

    def reset(self) -> None:
        self.curr_user = self.user_state._generate_observable_state()
        self.is_terminal = False

    def render(self):
        raise NotImplementedError()

    def get_candidate_docs(self) -> torch.Tensor:
        self.candidate_docs = torch.stack(self.user_state._user_candidate_items())
        return self.candidate_docs.to(self.device)

    def get_clicked_docs(self) -> torch.Tensor:
        self.clicked_docs = torch.stack(self.user_state.selected_item_feature())
        return self.clicked_docs

    def hidden_state(self):
        self.hidden_choice_state = self.user_state._generate_hidden_state()
        return self.hidden_choice_state

    def diversity(self):
        self.diversity_value = self.user_state.entropy_based_diversity()
        return self.diversity_value

    def modified_llm_response(self, slate_features, client):
        if client is None:
            return slate_features

        user_history = self.user_state.get_click_history()
        candidate_docs, slate_docs = self.user_state.get_llm_ids(
            self.candidate_docs, slate_features
        )
        user_history = self.user_state.get_item_ids_and_titles(user_history)
        candidate_titles = self.user_state.get_item_ids_and_titles(candidate_docs)
        slate_titles = self.user_state.get_item_ids_and_titles(slate_docs)

        prompt = f"""
        You are a slate generator. Given a user's interaction history, select the best 10 items to recommend.

        User History:
        {', '.join([f"{item} ({title})" for item, title in user_history])}

        Candidate Docs:
        {', '.join([f"{item} ({title})" for item, title in candidate_titles])}

        Given Slate:
        {', '.join([f"{item} ({title})" for item, title in slate_titles])}

        What changes will you make to this slate? If you think any item needs to be changed, consider selecting it from the candidate docs. Take this decision in such a way that the user's click probability from the slate is maximized. The user is only allowed to pick one item.

        Please provide the recommended slate as a numbered list of 10 items. Please keep it short and concise.
        """

        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            response_text = (
                response.choices[0].message.content if response.choices else None
            )
        except Exception as e:
            self.logger.error(f"Error processing user {self.curr_user}: {str(e)}")
            response_text = None

        self.row_counter += 1
        self.log_recommendation(slate_titles, response_text)

        recommended_slate = self.parse_recommended_slate(response_text)
        if recommended_slate:
            llm_slate = torch.stack(
                self.user_state.get_llm_features(recommended_slate)
            ).to(self.device)
            if len(llm_slate) < 10:
                num_needed = 10 - len(llm_slate)
                # Take first 'num_needed' items from original slate to maintain order
                padding_features = slate_features[:num_needed, :]
                llm_slate = torch.cat([llm_slate, padding_features])
                return llm_slate
        slate_features = slate_features[:10]
        return slate_features

    def setup_logger(self):
        # Only create new file if we don't have one or need rotation
        if self.current_log_file is None or (self.row_counter - 1) % 50 == 0:
            self.current_log_file = (
                self.LOG_DIR
                / f"recommendation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            logger = logging.getLogger(f"slate_logger_{self.current_log_file.stem}")

            # Remove existing handlers if any
            if logger.handlers:
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)

            file_handler = logging.FileHandler(self.current_log_file)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            logger.addHandler(file_handler)

            error_handler = logging.FileHandler(self.LOG_DIR / "error_log.log")
            error_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            logger.addHandler(error_handler)

            logger.setLevel(logging.INFO)
            return logger
        return self.logger

    def log_recommendation(self, slate_titles, response_text):
        # Just ensure logger is set up properly
        self.logger = self.setup_logger()

        self.logger.info(f"Initial User State: {self.curr_user}")
        self.logger.info("Given Slate Items:")
        for item_id, title in slate_titles:
            self.logger.info(f"  {item_id}: {title}")
        self.logger.info(f"LLM Response:\n{response_text}")
        self.logger.info("-" * 50)

    def parse_recommended_slate(self, response_text):
        if response_text is None:
            return []
        else:
            item_id_pattern = r"N\d+"

            # Find all matches of the pattern in the response text
            item_ids = re.findall(item_id_pattern, response_text)
            # Remove single quotes from item IDs (if any)
            item_ids = [item_id.replace("'", "") for item_id in item_ids]

            # Remove duplicates while preserving order
            item_ids = list(dict.fromkeys(item_ids))

            return item_ids
