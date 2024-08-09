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

def device():
    """環境毎に利用できるアクセラレータを返す"""
    if torch.backends.mps.is_available():
        # macOS w/ Apple Silicon or AMD GPU
        return "mps"
    if torch.cuda.is_available():
        # NVIDIA GPU
        return "cuda"
    return "cpu"

DEVICE = device()
print(f"device: {DEVICE}")

# Hyper parameter
wordvec_size = 100
hidden_size = 100
batch_size = 10
T = 20
learning_rate = 0.01
epochs = 100

#Embedding

class Embedding(nn.Module):
    def __init__(self, vocab_size, wordvec_size):
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, wordvec_size))
    
    def forward(self, id):
        W = self.weight
        self.id = id
        out = W[id]
        return out
    
class TimeEmbedding(nn.Module):
    def __init__(self, vocab_size, wordvec_size, T):
        super(TimeEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.wordvec_size = wordvec_size
        self.T = T
        self.embedding = Embedding(self.vocab_size, self.wordvec_size)
    
    def forward(self, idx):
        batch_size, sequence_length = idx.shape
        out = torch.empty(batch_size, self.T, self.wordvec_size)

        for t in range(self.T):
            out[:, t, :] = self.embedding(idx[:, t])
        
        return out

class TimeAffine(nn.Module):
    def __init__(self, hidden_size, vocab_size, T):
        super(TimeAffine, self).__init__()
        self.affine = nn.Linear(hidden_size, vocab_size)
        self.T = T

    def forward(self, h_results):
        output = torch.empty_like(h_results)
        for t in range(self.T):
            output[:, t, :] = self.affine(h_results[:, t, :])
        return output

# LSTMCell definition
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size): # 初期化
        super(LSTMCell, self).__init__()
        self.linear_x = nn.Linear(input_size, 4*hidden_size, bias=False)
        self.linear_h = nn.Linear(hidden_size, 4*hidden_size, bias=False)
        self.bias = nn.Parameter(torch.randn(4*hidden_size))

    def forward(self, x, h, c): # 順伝播
        x = self.linear_x(x)
        h = self.linear_h(h)
        A = torch.tanh(x + h + self.bias)

        f = A[:, :hidden_size]
        g = A[:, hidden_size: 2*hidden_size]
        i = A[:, 2*hidden_size: 3*hidden_size]
        o = A[:, 3*hidden_size:]

        f = F.sigmoid(f)
        g = F.tanh(g)
        i = F.sigmoid(i)
        o = F.sigmoid(o)
        
        c_next = f * c + g * i
        h_next = o * F.tanh(c)
        return h_next, c_next

# TimeLSTM definition
class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, T, stateful=False):
        super(TimeLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.T = T
        self.layer = LSTMCell(input_size, hidden_size)
        self.stateful = stateful
    
    def forward(self, X, hs=None,cs=None):
        if not self.stateful or hs is None or cs is None:
            h = torch.zeros((self.batch_size, self.hidden_size))
            c = torch.zeros((self.batch_size, self.hidden_size))
        else:
            h = hs
            c = cs

        h_results = torch.empty((self.batch_size, self.T, self.hidden_size))
        for t in range(self.T):
            h_results[:, t, :] = h
            h, c = self.layer(X[:, t, :], h, c)

        return h_results

class simpleLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, wordvec_size, batch_size, T, dropout=True):
        super(simpleLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.wordvec_size = wordvec_size
        self.batch_size = batch_size
        self.T = T
        self.dropout = dropout
        
        self.time_embedding = TimeEmbedding(self.vocab_size, self.wordvec_size, self.T)
        self.time_LSTM_1 = TimeLSTM(self.wordvec_size, self.hidden_size, self.batch_size, self.T)
        self.time_LSTM_2 = TimeLSTM(self.hidden_size, self.hidden_size, self.batch_size, self.T)
        self.time_affine = TimeAffine(self.hidden_size, self.vocab_size, self.T)
        
        if self.dropout:
            self.dropout_1 = nn.Dropout()
            self.dropout_2 = nn.Dropout()
            self.dropout_3 = nn.Dropout()
        
    def forward(self, idx, hs=None):
        X = self.time_embedding(idx)
        if self.dropout:
            X = self.dropout_1(X)
        H = self.time_LSTM_1(X, hs)
        if self.dropout:
            H = self.dropout_2(H)
        H = self.time_LSTM_2(X, hs)
        if self.dropout:
            H = self.dropout_3(H)
        V = self.time_affine(H)
        output = F.softmax(V)
        return output

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
corpus_size = 1001
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1]  # input
ts = corpus[1:]   # output (teaching label)

data_size = len(xs)
print('corpus size: %d, vocabulary size: %d' % (corpus_size, vocab_size))

# Create dataset and dataloader
dataset = PTBDataset(xs, ts, T)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Create model
model = simpleLSTM(vocab_size, hidden_size, wordvec_size, batch_size, T).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
losses = []

# Training loop
for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    for x, label in dataloader:
        x = x.to(DEVICE)
        label = label.to(DEVICE)
        # テンソルに変換
        ts_tensor = torch.tensor(label, dtype=torch.long)
        # ワンホットエンコーディング
        ts_one_hot = F.one_hot(ts_tensor, num_classes=vocab_size)
        # float型に変換
        ts_one_hot_float = ts_one_hot.float()

        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, ts_one_hot_float)
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
