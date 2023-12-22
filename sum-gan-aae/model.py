import torch
import torch.nn as nn
from torch.autograd import Variable

from util import StackedLSTMCell

# summarizer LSTM
# input: video features - [n_frames, n_feats_per_frame]
# output: weights/importance of each frame - [n_frames]
class sLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers=2):
    super().__init__()

    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
    self.out = nn.Sequential(
      nn.Linear(hidden_size * 2, 1),
      nn.Sigmoid()
    )

  def forward(self, feats):
    self.lstm.flatten_parameters()
    feats, (h_n, c_n) = self.lstm(feats)

    scores = self.out(feats.squeeze(1))
    return scores


# encoder LSTM
# input: video features - [n_frames, 1, hidden_size]
# output: last hidden - h_last [num_layers=2, 1, hidden_size], c_last [num_layers =2, 1, hidden_size]
class eLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers=2):
    super().__init__()

    self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
    self.linear_mu = nn.Linear(hidden_size, hidden_size)
    self.linear_var = nn.Linear(hidden_size, hidden_size)

  def forward(self, feats):
    self.lstm.flatten_parameters()
    _, (h_last, c_last) = self.lstm(feats)

    return (h_last, c_last)


# decoder LSTM
# input: seq_len [n_frames], init_hidden: h [n_layers, 1, hidden_size], c [n_layers, 1, hidden_size]
# output: out_features [n_frames, 1, hidden_size]
class dLSTM(nn.Module):
  def __init__(self, input_size=2048, hidden_size=2048, num_layers=2,
              device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    super().__init__()
    self.device = device

    self.lstm_cell = StackedLSTMCell(num_layers, input_size, hidden_size)
    self.out = nn.Linear(hidden_size, input_size)

  def forward(self, seq_len, init_hidden):
    batch_size = init_hidden[0].size(1)
    hidden_size = init_hidden[0].size(2)

    x = torch.zeros(batch_size, hidden_size).to(self.device)
    h, c = init_hidden  # (h0,  c0): last state of eLSTM

    out_feats = []
    for _ in range(seq_len):
      (last_h, last_c), (h, c) = self.lstm_cell(x, (h, c))
      x = self.out(last_h)
      out_feats.append(last_h)

    return out_feats


# input: features [n_frames, 1, hidden_size]
# output: h [num_layers=2, 1, hidden_size], decoded_features [n_frames, 1, n_feats=2048]
class VAE(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers=2, 
              device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    super().__init__()
    self.device = device

    self.e_lstm = eLSTM(input_size, hidden_size, num_layers)
    self.d_lstm = dLSTM(input_size, hidden_size, num_layers)

    self.softplus = nn.Softplus()

  # sample z via reparameterization trick
  def reparameterize(self, mu, log_variance):
    std = torch.exp(0.5 * log_variance)
    epsilon = torch.randn(std.size()).to(self.device) # e ~ N(0,1)

    # [n_layers, 1, hidden_size]
    return (mu + epsilon * std).unsqueeze(1)

  def forward(self, feats):
    seq_len = feats.size(0)
    h, c = self.e_lstm(feats)
    h = h.squeeze(1)

    h_mu = self.e_lstm.linear_mu(h)
    h_log_variance = torch.log(self.softplus(self.e_lstm.linear_var(h)))

    h = self.reparameterize(h_mu, h_log_variance)

    decoded_feats = self.d_lstm(seq_len, init_hidden=(h, c))
    decoded_feats.reverse() # [n_frames, 1, hidden_size]
    decoded_feats = torch.stack(decoded_feats)

    return h_mu, h_log_variance, decoded_feats


class Summarizer(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers=2):
    super().__init__()
    self.s_lstm = sLSTM(input_size, hidden_size, num_layers)
    self.vae = VAE(input_size, hidden_size, num_layers)

  def forward(self, img_feats, uniform=False):
    if not uniform:
      scores = self.s_lstm(img_feats)
      weighted_feats = img_feats * scores.view(-1, 1, 1)
    else:
      scores = None
      weighted_feats = img_feats
    
    h_mu, h_log_variance, decoded_feats = self.vae(weighted_feats)
    
    return scores, h_mu, h_log_variance, decoded_feats


# input: features [n_frames, 1, input_size]
# output: h_n, c_n [n_layeres * n_direction, batch_size * num_directions]
class cLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers=2):
    super().__init__()

    self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

  def forward(self, feats, init_hidden=None):
    out, (h_n, c_n) = self.lstm(feats, init_hidden)
    last_h = h_n[-1]  # [batch_size, hidden_state]
    return last_h


# input: features [n_frames, 1, hidden_size]
# output: h [1, hidden_size] (last h from top layer of discriminator)
#         prob [batch_size=1, 1] (probability to be original feature from CNN)
class Discriminator(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers=2):
    super().__init__()

    self.cLSTM = cLSTM(input_size, hidden_size, num_layers)
    self.out = nn.Sequential(
      nn.Linear(hidden_size, 1),
      nn.Sigmoid()
    )

  def forward(self, feats):
    h = self.cLSTM(feats)
    prob = self.out(h).squeeze()

    return h, prob
