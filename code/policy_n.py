import torch
import torch.nn as nn
# Define the policy network π(s)   φ
class PolicyNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2, init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z)
        distribution = torch.distributions.Normal(mean, std)
        log_prob = distribution.log_prob(mean + std * z) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std

    def generate_action(self, state):
        '''
        Sampling Actions: The policy network in SAC is a stochastic policy that outputs a probability distribution
        over actions for a given state. During training and exploration, actions are sampled from this probability
        distribution. This stochasticity allows the agent to explore different actions and
        promotes better exploration of the environment.
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z)

        action = action.cpu()
        return action[0]

