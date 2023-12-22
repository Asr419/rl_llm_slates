import pytest
from rl_mind_dataset.document_modelling.data_reader import DatasetReader


def test_get_category_list():
    dataset_reader = DatasetReader()
    category_list = dataset_reader.get_category_list()
    a = 5
