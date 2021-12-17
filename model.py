import torch
import torch.nn as nn

class DeepSDF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(259, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512 - 259)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, 1)

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

