import torch
from torch import nn

try:
    from torch_geometric.nn import GATv2Conv
except ModuleNotFoundError as exc:
    raise ImportError(
        "BOPOFJSPModel requires torch_geometric. Install the BOPO/FJSP dependency set before running this baseline."
    ) from exc


class CAMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        embed_size: int = 128,
        dropout: float = 0.0,
        leaky_slope: float = 0.15,
    ):
        super().__init__()
        self.ops_gat1 = GATv2Conv(
            in_channels=input_size,
            out_channels=hidden_size,
            dropout=dropout,
            concat=False,
            heads=2,
            add_self_loops=False,
            negative_slope=leaky_slope,
        )
        self.ops_gat2 = GATv2Conv(
            in_channels=hidden_size * 3 + input_size,
            out_channels=embed_size,
            dropout=dropout,
            concat=False,
            heads=2,
            add_self_loops=False,
            negative_slope=leaky_slope,
        )
        self.job_gat1 = GATv2Conv(
            in_channels=input_size,
            out_channels=hidden_size,
            dropout=dropout,
            concat=False,
            heads=2,
            add_self_loops=False,
            negative_slope=leaky_slope,
        )
        self.job_gat2 = GATv2Conv(
            in_channels=hidden_size * 3 + input_size,
            out_channels=embed_size,
            dropout=dropout,
            concat=False,
            heads=2,
            add_self_loops=False,
            negative_slope=leaky_slope,
        )
        self.mac_gat1 = GATv2Conv(
            in_channels=input_size,
            out_channels=hidden_size,
            dropout=dropout,
            concat=False,
            heads=2,
            add_self_loops=False,
            negative_slope=leaky_slope,
        )
        self.mac_gat2 = GATv2Conv(
            in_channels=hidden_size * 3 + input_size,
            out_channels=embed_size,
            dropout=dropout,
            concat=False,
            heads=2,
            add_self_loops=False,
            negative_slope=leaky_slope,
        )
        self.linear = nn.Linear(embed_size * 3, embed_size)
        self.out_size = input_size + embed_size

    def forward(
        self,
        x: torch.Tensor,
        ops_egdes: torch.Tensor,
        job_edges: torch.Tensor,
        mac_edges: torch.Tensor,
    ) -> torch.Tensor:
        h1_ops = torch.relu(self.ops_gat1(x, ops_egdes))
        h1_job = torch.relu(self.job_gat1(x, job_edges))
        h1_mac = torch.relu(self.mac_gat1(x, mac_edges))
        h = torch.cat([x, h1_job, h1_mac, h1_ops], dim=-1)

        h2_ops = torch.relu(self.ops_gat2(h, ops_egdes))
        h2_job = torch.relu(self.job_gat2(h, job_edges))
        h2_mac = torch.relu(self.mac_gat2(h, mac_edges))
        h2 = torch.relu(self.linear(torch.cat([h2_ops, h2_job, h2_mac], dim=-1)))
        return torch.cat([x, h2], dim=-1)


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        encoder_size: int,
        context_size: int,
        hidden_size: int = 64,
        att_size: int = 128,
        leaky_slope: float = 0.15,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.act = nn.LeakyReLU(leaky_slope)
        self.norm = nn.LayerNorm(att_size)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(encoder_size, att_size)
        self.lstm = nn.LSTM(att_size, att_size, batch_first=True)
        self.linear2 = nn.Linear(att_size, att_size, bias=False)
        self.linear3 = nn.Linear(context_size, hidden_size)
        self.linear4 = nn.Linear(context_size, hidden_size)
        self.linear5 = nn.Linear(hidden_size * 2 + encoder_size, att_size, bias=False)
        self._init_weight()

    def forward(self, embed, contexts, last_embed, h=None, c=None):
        x = self.act(self.linear1(last_embed))
        if h is None:
            state, (new_h, new_c) = self.lstm(x)
        else:
            state, (new_h, new_c) = self.lstm(x, (h, c))
        query = self.linear2(self.norm(self.dropout(state) + x))

        context_j, context_m = contexts
        score_j = self.act(self.linear3(context_j))
        score_m = self.act(self.linear4(context_m))
        key = self.linear5(torch.cat([embed, score_j, score_m], dim=-1))
        logits = torch.bmm(key, query.permute(0, 2, 1)).squeeze(-1)
        return logits, (new_h, new_c)

    def _init_weight(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param, gain=1.0)
        nn.init.xavier_uniform_(self.linear1.weight.data, gain=0.25)
        nn.init.xavier_uniform_(self.linear2.weight.data, gain=0.25)
        nn.init.xavier_uniform_(self.linear3.weight.data, gain=0.25)
        nn.init.xavier_uniform_(self.linear4.weight.data, gain=0.25)
