import torch
import torch.nn as nn

from constants import *

class LatentEmbedding(nn.Module):
    """
    A LUT that converts mesh index to a latent vector
    """
    def __init__(self, num_models):
        super().__init__()
        self.embedding = nn.Embedding(num_models, LATENT_DIM)
        torch.nn.init.uniform_(self.embedding.weight, LATENT_MU, LATENT_SIGMA)
    
    def forward(self, x):
        return self.embedding(x)


class DeepSDF(nn.Module):
    """
    The DeepSDF neural network, as presented in the original paper
    https://arxiv.org/abs/1901.05103
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.utils.weight_norm(nn.Linear(LATENT_DIM + 3, HIDDEN_DIM))
        self.fc2 = nn.utils.weight_norm(nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
        self.fc3 = nn.utils.weight_norm(nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
        self.fc4 = nn.utils.weight_norm(nn.Linear(HIDDEN_DIM, HIDDEN_DIM - LATENT_DIM - 3))
        self.fc5 = nn.utils.weight_norm(nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
        self.fc6 = nn.utils.weight_norm(nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
        self.fc7 = nn.utils.weight_norm(nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
        self.fc8 = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, xyz, latent):
        input = torch.cat((xyz, latent), dim=1)
        x = torch.relu(self.fc1(input))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.cat((x, input), dim=1)
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.tanh(self.fc8(x))
        return x

