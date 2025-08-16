import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PositionEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        position = torch.arange(0, max_seq_len).unsqueeze(1)

        item = 1/10000**(torch.arange(0, d_model, 2) / d_model)

        tmp_pos = position * item
        pe = torch.zeros(max_seq_len, d_model)

        pe[:, 0::2] = torch.sin(tmp_pos)
        pe[:, 1::2] = torch.cos(tmp_pos)

        plt.matshow(pe)
        plt.show()

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, False)

    def forwoard(self, x):
        batch, seq_len, _ = x.shape
        pe = self.pe
        return x + pe[:, :seq_len, :]
    
if __name__ == '__main__':
    PositionEncoding(512, 100)