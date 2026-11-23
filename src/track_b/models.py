"""RNN 계열 예측기: LSTM / GRU / (보너스) Attention-LSTM."""
import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    """Bahdanau-style: score_t = vᵀ tanh(W h_t), α = softmax(score), context = Σ α_t h_t"""

    def __init__(self, hidden):
        super().__init__()
        self.W = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, h):  # h: (B, L, H)
        a = torch.softmax(self.v(torch.tanh(self.W(h))).squeeze(-1), dim=1)  # (B, L)
        return (a.unsqueeze(-1) * h).sum(1), a


class RNNForecaster(nn.Module):
    def __init__(self, cell="lstm", input_size=1, hidden=64, layers=2, dropout=0.2, attention=False):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(input_size, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.attn = AdditiveAttention(hidden) if attention else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        h, _ = self.rnn(x)
        if self.attn is not None:
            ctx, a = self.attn(h)
        else:
            ctx, a = h[:, -1], None
        return self.head(ctx).squeeze(-1), a


MODEL_ZOO = {
    "LSTM": dict(cell="lstm"),
    "GRU": dict(cell="gru"),
    "Attn-LSTM": dict(cell="lstm", attention=True),
}
