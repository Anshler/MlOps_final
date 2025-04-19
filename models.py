import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, hidden_layer=64):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(20, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, hidden_layer)
        self.fc4 = nn.Linear(hidden_layer, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x