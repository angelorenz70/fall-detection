import torch.nn as nn

class BiLSTM(nn.Module):
    # def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
    #     super(BiLSTM, self).__init__()
    #     self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
    #     self.fc = nn.Linear(2*hidden_dim, num_classes)
    #     self.hidden_dim = hidden_dim

    # def forward(self, x):
    #     lstm_out, _ = self.lstm(x)
    #     lstm_out = lstm_out.view(-1, 2 * self.hidden_dim)
    #     output = self.fc(lstm_out)  # Take the last time step's output
    #     return output
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, sequence_length):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim * sequence_length, num_classes)  # Adjust the linear layer input size
        self.hidden_dim = hidden_dim

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Flatten the LSTM output to pass through the linear layer
        lstm_out = lstm_out.view(-1, 2 * self.hidden_dim * x.size(1))
        return lstm_out
