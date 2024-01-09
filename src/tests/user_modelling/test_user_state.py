import pytest
from rl_mind_dataset.document_modelling.data_reader import DatasetReader

from rl_mind_dataset.user_modelling.user_state import UserState


def test_generate_observable_state():
    user_state = UserState()
    user_state._generate_observable_state()
    print(f"Observable State: {user_state.user_state}")
    assert user_state.user_state is not None


def test_generate_hidden_state():
    user_state = UserState()
    user_state._generate_hidden_state()
    print(f"Hidden State Length: {len(user_state.hidden_user_state)}")
    assert len(user_state.hidden_user_state) == 18
