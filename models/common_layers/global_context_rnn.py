import torch
import torch.nn as nn


class GlobalContextRNN(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = .3
    ) -> None:
        super().__init__()

        self.gru_global_context = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True
        )

        self.out = nn.Linear(hidden_dim, output_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
