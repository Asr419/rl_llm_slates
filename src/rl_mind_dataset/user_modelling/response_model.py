import abc
from typing import Any

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
        doc_repr: torch.Tensor,
        clicked_doc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        satisfaction = torch.dot(estimated_user_state, doc_repr)
        doc_quality = torch.dot(doc_repr, clicked_doc)
        response = (1 - self.alpha) * satisfaction + self.alpha * doc_quality
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
