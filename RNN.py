import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


T=5

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        # self.b = 0
        self.linear_x = nn.Linear(input_size, hidden_size, bias=False)
        self.linear_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        x = self.linear_x(x)
        h = self.linear_h(h)
        h_next = torch.tanh(x+h)

        return h_next

class TimeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, stateful = False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.layer= RNNCell(input_size, hidden_size)
        self.stateful = stateful
    
    def forward(self, X, hs):
        if not self.stateful:
            h = torch.zeros((self.hidden_size, self.batch_size))
        else:
            h = hs

        h_results = torch.empty((hs.shape(0), T, hs.shape(1)))
        for t in range(T):
            h = RNNCell(X[:, t, :], h)
            h_results[:, t, :] = h

        return h_results





