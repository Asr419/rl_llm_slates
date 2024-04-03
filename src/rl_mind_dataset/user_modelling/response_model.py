import abc
from typing import Any
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch


class AbstractResponseModel(metaclass=abc.ABCMeta):
    def __init__(self, null_response: float = -1.0) -> None:
        self.null_response = null_response

    @abc.abstractmethod
    def generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate the user response (reward) to a slate,
        is a function of the user state and the chosen document in the slate.

        Args:
            estimated_user_state (np.array): estimated user state
            doc_repr (np.array): document representation

        Returns:
            float: user response
        """
        pass

    def generate_null_response(self) -> torch.Tensor:
        return torch.tensor(self.null_response)


class AmplifiedResponseModel(AbstractResponseModel):
    def __init__(self, amp_factor: int = 1, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.amp_factor = amp_factor

    @abc.abstractmethod
    def _generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        pass

    def generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return (
            self._generate_response(estimated_user_state, doc_repr, **kwargs)
            * self.amp_factor
        )

    def generate_null_response(self) -> float:
        return super().generate_null_response() * self.amp_factor


class WeightedDotProductResponseModel:
    def __init__(
        self,
        amp_factor: int = 1,
        alpha: float = 1.0,
        device: torch.device = torch.device("cpu"),
        null_response: float = -1.0,
        **kwds: Any,
    ) -> None:
        super().__init__(**kwds)
        self.amp_factor = amp_factor
        self.alpha = alpha
        self.null_response = null_response

    def _generate_response(
        self,
        estimated_user_state: torch.Tensor,
        selected_doc: torch.Tensor,
        actual_clicked_doc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        satisfaction = torch.dot(estimated_user_state, selected_doc)
        doc_quality = torch.dot(selected_doc, actual_clicked_doc)

        response = (1 - self.alpha) * satisfaction + self.alpha * doc_quality
        # response = doc_quality
        return response

    def generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return (
            self._generate_response(estimated_user_state, doc_repr, **kwargs)
            * self.amp_factor
        )

    def generate_null_response(self) -> torch.Tensor:
        return torch.tensor(self.null_response)


class WeightedCosineResponseModel:
    def __init__(
        self,
        amp_factor: int = 1,
        alpha: float = 1.0,
        device: torch.device = torch.device("cpu"),
        null_response: float = -1.0,
        **kwds: Any,
    ) -> None:
        super().__init__(**kwds)
        self.amp_factor = amp_factor
        self.alpha = alpha
        self.null_response = null_response

    def _generate_response(
        self,
        estimated_user_state: torch.Tensor,
        selected_doc: torch.Tensor,
        actual_clicked_doc: torch.Tensor,
        diversity: torch.Tensor,
        alpha: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        satisfaction = torch.nn.functional.cosine_similarity(
            estimated_user_state, selected_doc, dim=0
        )
        doc_quality = torch.nn.functional.cosine_similarity(
            selected_doc, actual_clicked_doc, dim=0
        )
        response = (1 - alpha) * doc_quality + alpha * (1 - satisfaction)
        # response = (1 - self.alpha) * satisfaction + self.alpha * doc_quality
        # response = satisfaction
        return response, satisfaction, doc_quality

    def generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return (
            self._generate_response(estimated_user_state, doc_repr, **kwargs)
            * self.amp_factor
        )

    def generate_null_response(self) -> torch.Tensor:
        return torch.tensor(self.null_response)


class WeightedUserNCFResponseModel:
    def __init__(
        self,
        amp_factor: int = 1,
        alpha: float = 1.0,
        device: torch.device = torch.device("cpu"),
        null_response: float = -1.0,
        **kwds: Any,
    ) -> None:
        super().__init__(**kwds)
        self.amp_factor = amp_factor
        self.alpha = alpha
        self.null_response = null_response
        self.device = device
        base_path = Path.home() / Path(os.environ.get("SAVE_PATH"))
        RUN_BASE_PATH = Path(f"user_choice_model_1")
        PATH = base_path / RUN_BASE_PATH / Path("model.pt")
        self.model = torch.load(PATH).to(self.device)

    def _generate_response(
        self,
        estimated_user_state: torch.Tensor,
        selected_doc: torch.Tensor,
        actual_clicked_doc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        satisfaction = torch.sigmoid(
            self.model(
                estimated_user_state, selected_doc, train=False, device=self.device
            )
        )

        doc_quality = torch.sigmoid(
            self.model(
                selected_doc, actual_clicked_doc, train=False, device=self.device
            )
        )

        response = (1 - self.alpha) * satisfaction + self.alpha * doc_quality
        # response = doc_quality
        return response

    def generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return (
            self._generate_response(estimated_user_state, doc_repr, **kwargs)
            * self.amp_factor
        )

    def generate_null_response(self) -> torch.Tensor:
        return torch.tensor(self.null_response)


class CosineResponseModel(AmplifiedResponseModel):
    def _generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
    ) -> torch.Tensor:
        satisfaction = torch.nn.functional.cosine_similarity(
            estimated_user_state, doc_repr, dim=0
        )
        return satisfaction


class DotProductResponseModel(AmplifiedResponseModel):
    def _generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
    ) -> torch.Tensor:
        satisfaction = torch.dot(estimated_user_state, doc_repr)
        return satisfaction
