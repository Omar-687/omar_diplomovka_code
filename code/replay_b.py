import random
import torch
import torch.nn as nn

class ReplayBuffer(nn.Module):

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):

        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
