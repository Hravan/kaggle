import torch

from torch import nn

class RNNGRU(nn.Module):
    '''A base class for recurrent neural networks with GRU unit.'''

    def __init__(self, n_vocab, emb_dim, hidden_size, n_layers=1):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_size, num_layers=n_layers)
        self.out = nn.Linear(hidden_size, 1)

class RNNGRUSequential(RNNGRU):

    def forward(self, x):
        x1, x2 = x
        x1_emb = self.emb(x1)
        x2_emb = self.emb(x2)

        concatenated = torch.cat([x1_emb, x2_emb])
        _, x = self.gru(concatenated)

        out = torch.sigmoid(self.out(x))
        return out.view(-1)

class RNNGRUParallel(RNNGRU):

    def forward(self, x):
        x1, x2 = x
        x1_emb = self.emb(x1)
        x2_emb = self.emb(x2)

        _, x1_gru_out = self.gru(x1_emb)
        _, x2_gru_out = self.gru(x2_emb)

        subtracted_abs = torch.abs(x1_gru_out[-1] - x2_gru_out[-1])

        out = torch.sigmoid(self.out(subtracted_abs))
        return out.view(-1)
