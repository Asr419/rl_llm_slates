import pytest
from rl_mind_dataset.document_modelling.data_reader import DatasetReader


def test_get_category_list():
    dataset_reader = DatasetReader()
    df = dataset_reader.processed_slate_datatset()
    assert len(df) == 65278
