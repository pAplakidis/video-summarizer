import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

W, H = 224, 224
N_EPOCHS = 300
LR = 1e-4
INPUT_SIZE = 1024
HIDDEN_SIZE = 500
NUM_LAYERS = 2
SUMMARY_RATE = 0.3
CLIP = 5.0
DISCRIMINATOR_SLOW_START = 15
THRESHOLD = 0.3


class TensorboardWriter(SummaryWriter):
    def __init__(self, logdir):
        """
        Extended SummaryWriter Class from tensorboard-pytorch (tensorbaordX)
        https://github.com/lanpa/tensorboard-pytorch/blob/master/tensorboardX/writer.py
        Internally calls self.file_writer
        """
        super(TensorboardWriter, self).__init__(logdir)
        self.logdir = self.file_writer.get_logdir()

    def update_parameters(self, module, step_i):
        """
        module: nn.Module
        """
        for name, param in module.named_parameters():
            self.add_histogram(name, param.clone().cpu().data.numpy(), step_i)

    def update_loss(self, loss, step_i, name='loss'):
        self.add_scalar(name, loss, step_i)

    def update_histogram(self, values, step_i, name='hist'):
        self.add_histogram(name, values, step_i)

class StackedLSTMCell(nn.Module):
  def __init__(self, num_layers, input_size, rnn_size, dropout=0.0):
    super(StackedLSTMCell, self).__init__()
    self.dropout = nn.Dropout(dropout)
    self.num_layers = num_layers
    self.layers = nn.ModuleList()

    for i in range(num_layers):
      self.layers.append(nn.LSTMCell(input_size, rnn_size))
      input_size = rnn_size

  def forward(self, x, h_c):
    """
    Args:
        x: [batch_size, input_size]
        h_c: [2, num_layers, batch_size, hidden_size]
    Return:
        last_h_c: [2, batch_size, hidden_size] (h from last layer)
        h_c_list: [2, num_layers, batch_size, hidden_size] (h and c from all layers)
    """
    h_0, c_0 = h_c
    h_list, c_list = [], []
    for i, layer in enumerate(self.layers):
      # h of i-th layer
      h_i, c_i = layer(x, (h_0[i], c_0[i]))

      # x for next layer
      x = h_i
      if i + 1 != self.num_layers:
          x = self.dropout(x)
      h_list += [h_i]
      c_list += [c_i]

    last_h_c = (h_list[-1], c_list[-1])
    h_list = torch.stack(h_list)
    c_list = torch.stack(c_list)
    h_c_list = (h_list, c_list)

    return last_h_c, h_c_list

def get_laplacian_scores(frames):
  variance_laplacians = []

  for f in frames:
    f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    variance_laplacian = cv2.Laplacian(f, cv2.CV_64F).var()
    variance_laplacians.append(variance_laplacian)

  return variance_laplacians
