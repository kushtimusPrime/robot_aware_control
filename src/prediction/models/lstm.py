import torch
import torch.nn as nn
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class LSTM(nn.Module):
    """
    Vanilla LSTM with embedding layer and output
    """

    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList(
            [nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)]
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Tanh(),
        )
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        hidden = []
        for i in range(self.n_layers):
            hidden.append(
                (
                    Variable(torch.zeros(batch_size, self.hidden_size).to(device)),
                    Variable(torch.zeros(batch_size, self.hidden_size).to(device)),
                )
            )
        return hidden

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        # pass it through the LSTM cells, and cache hidden states for future
        # forward calls.
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        return self.output(h_in)


class GaussianLSTM(nn.Module):
    """
    Outputs latent mean and std P(z | x)
    """

    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(GaussianLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList(
            [nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)]
        )
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        hidden = []
        for i in range(self.n_layers):
            hidden.append(
                (
                    Variable(torch.zeros(batch_size, self.hidden_size).to(device)),
                    Variable(torch.zeros(batch_size, self.hidden_size).to(device)),
                )
            )
        return hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


if __name__ == "__main__":
    glstm = GaussianLSTM(10, 10, 32, 2, 64)
