import torch
import torch.nn as nn


    # Define the soft Q-function network Q(s, a)
#     θ
class SoftQNetwork(nn.Module):

    def __init__(self, state_dim, hidden_dim=256):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x






