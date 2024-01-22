import os

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class DatasetReader:
    _DATA_PATH = Path.home() / Path(os.environ.get("DATA_PATH"))
    USER_INTERACTION = Path("MINDlarge_train/behaviors.tsv")
    NEWS = Path("MINDlarge_train/news.tsv")

    def __init__(self) -> None:
        pass

    def get_data_path(self) -> Path:
        return self._DATA_PATH

    def get_articles_dataset(self) -> pd.DataFrame:
        path = self.get_data_path() / self.NEWS
        news_df = pd.read_csv(
            path,
            sep="\t",
            names=[
                "itemId",
                "category",
                "subcategory",
                "title",
                "abstract",
                "url",
                "title_entities",
                "abstract_entities",
            ],
        )
        return news_df

    def get_user_interaction(self) -> pd.DataFrame:
        path = self.get_data_path() / self.USER_INTERACTION
        user_interaction_df = pd.read_csv(
            path,
            sep="\t",
            names=[
                "impressionId",
                "userId",
                "timestamp",
                "click_history",
                "impressions",
            ],
        )
        return user_interaction_df

    def processed_slate_datatset(self) -> pd.DataFrame:
        df = pd.read_feather(
            self.get_data_path() / Path("MINDlarge_train/interaction_all_50.feather")
        )

        return df

    def get_category_list(self, row):
        article_category = self.article_category_map()
        if pd.notna(row["click_history"]):
            # Use a list comprehension to create the list of categories
            categories = [
                article_category.get(article_id, 0)
                for article_id in row["click_history"].split()
            ]
            count_categories = [categories.count(i) for i in range(0, 18)]
        else:
            # If 'click_history' is NaN, assign a list of zeros
            count_categories = []
        return count_categories

    def article_category_map(self):
        news = self.get_articles_dataset()
        news["num_category"] = news["category"].factorize()[0]
        article_category = news.set_index("itemId")["num_category"].to_dict()
        return article_category

    def item2vecdict(self):
        df = pd.read_feather(
            self.get_data_path() / Path("MINDlarge_train/news_glove_embed_50.feather")
        )
        embedding_dict = dict(zip(df["itemId"], df["embedding"]))
        embedding_dict = {k: v for k, v in embedding_dict.items() if v is not None}
        all_vectors = [np.array(vector) for vector in embedding_dict.values()]

        return embedding_dict, all_vectors
