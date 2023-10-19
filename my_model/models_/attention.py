import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        attn_weights = self.softmax(self.fc(x))
        output = torch.matmul(attn_weights, x)
        return output