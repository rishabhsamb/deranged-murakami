
import torch
from torch import nn

# LSTM implementation borrowed from Domas Bitvinskas
class LSTMModel(nn.Module):
    def __init__(self, dataset):
        super(LSTMModel, self).__init__()
        self.lstm_size = 256 # size of lstm input
        self.embedding_dim = 256 # dimensions in vector embedding of words
        self.num_layers = 8

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding( # embedding layer
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim
        )
        self.lstm = nn.LSTM( # simple lstm config
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers = self.num_layers,
            dropout=0.2
        )
        self.fc = nn.Linear(self.lstm, n_vocab) # output layer    
    def forward(self, x, prev_state):
        embed = self.embedding(x) # embed raw input
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output) 
        return logits, state
    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

