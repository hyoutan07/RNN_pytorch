import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import ptb

#Hyper parameter
wordvec_size = 100
hidden_size = 100
batch_size = 10
T=5
learning_rate=0.01
epochs = 100

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size): # 初期化
        # self.b = 0
        self.linear_x = nn.Linear(input_size, hidden_size, bias=False)
        self.linear_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h): #順伝播
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
            h_results[:, t, :] = h
            h = RNNCell(X[:, t, :], h)

        return h_results



model = TimeRNN(wordvec_size, hidden_size, batch_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
losses = []

# dataset loading (first 1000 words)
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1]  # input
ts = corpus[1:]   # output (teaching label)
data_size = len(xs)
print('corpus size: %d, vocabulary size: %d' % (corpus_size, vocab_size))


for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    for x, label in dataloader:
        optimizer.zero_grad()
        h_results = model(x, 0)
        loss = F.mse_loss(h_results, label)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    print(loss_avg)
    losses.append(loss_avg)

