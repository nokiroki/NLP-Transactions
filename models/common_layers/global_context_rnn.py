import torch
import torch.nn as nn


class GlobalContextRNN(nn.Module):

    def __init__(
        self
    ) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
