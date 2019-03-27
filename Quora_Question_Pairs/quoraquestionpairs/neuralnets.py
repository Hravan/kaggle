import torch

from torch import nn

class RNNGRU(nn.Module):
    '''A base class for recurrent neural networks with GRU unit.'''

    def __init__(self, n_vocab, emb_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

class RNNGRUSequential(RNNGRU):

    def forward(self, x1, x2):
        x1_emb = self.emb(x1)
        x2_emb = self.emb(x2)

        concatenated = torch.cat([x1_emb, x2_emb])
        _, x = self.gru(concatenated)

        out = torch.sigmoid(self.out(x))
        return out
