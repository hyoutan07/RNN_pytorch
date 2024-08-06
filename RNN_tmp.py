import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Dummy ptb module for example purpose
class ptb:
    @staticmethod
    def load_data(split):
        return np.random.randint(0, 100, 1000), {}, {}

# Hyper parameter
wordvec_size = 100
hidden_size = 100
batch_size = 10
T = 5
learning_rate = 0.01
epochs = 100

# RNNCell definition
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size): # 初期化
        super(RNNCell, self).__init__()
        self.linear_x = nn.Linear(input_size, hidden_size, bias=False)
        self.linear_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h): # 順伝播
        x = self.linear_x(x)
        h = self.linear_h(h)
        h_next = torch.tanh(x + h)
        return h_next

# TimeRNN definition
class TimeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, T, stateful=False):
        super(TimeRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.T = T
        self.layer = RNNCell(input_size, hidden_size)
        self.stateful = stateful
    
    def forward(self, X, hs):
        if not self.stateful:
            h = torch.zeros((self.batch_size, self.hidden_size))
        else:
            h = hs

        h_results = torch.empty((X.size(0), self.T, self.hidden_size))
        for t in range(self.T):
            h_results[:, t, :] = h
            h = self.layer(X[:, t, :], h)

        return h_results

# Dataset definition
class PTBDataset(Dataset):
    def __init__(self, xs, ts, T):
        self.xs = torch.tensor(xs, dtype=torch.long)
        self.ts = torch.tensor(ts, dtype=torch.long)
        self.T = T

    def __len__(self):
        return len(self.xs) - self.T

    def __getitem__(self, idx):
        x = self.xs[idx:idx + self.T]
        t = self.ts[idx:idx + self.T]
        return x, t

# Load dataset (first 1000 words)
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1]  # input
ts = corpus[1:]   # output (teaching label)
data_size = len(xs)
print('corpus size: %d, vocabulary size: %d' % (corpus_size, vocab_size))

# Create dataset and dataloader
dataset = PTBDataset(xs, ts, T)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create model
model = TimeRNN(wordvec_size, hidden_size, batch_size, T)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
losses = []

# Training loop
for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    for x, label in dataloader:
        optimizer.zero_grad()
        h_results = model(x.float(), None)
        loss = F.mse_loss(h_results, label.float())
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss_avg:.4f}')
    losses.append(loss_avg)

# Plot results
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
