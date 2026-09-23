# FROM https://github.com/Sachini/ronin/blob/master/source/model_temporal.py

"""
MIT License

Copyright (c) 2018 CMU Locus Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import os.path as osp

import torch
from torch.autograd import Variable

from comparison.source.ronin.tcn import TemporalConvNet


class LSTMSeqNetwork(torch.nn.Module):
    def __init__(self, input_size, out_size, device,
                 lstm_size=100, lstm_layers=3, dropout=0.2):
        """
        Simple LSTM network
        Input: torch array [batch x frames x input_size]
        Output: torch array [batch x frames x out_size]

        :param input_size: num. channels in input
        :param out_size: num. channels in output
        :param device: torch device
        :param lstm_size: number of LSTM units per layer
        :param lstm_layers: number of LSTM layers
        :param dropout: dropout probability of LSTM (@ref https://pytorch.org/docs/stable/nn.html#lstm)
        """
        super(LSTMSeqNetwork, self).__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.output_size = out_size
        self.num_layers = lstm_layers
        # self.batch_size = batch_size Useless
        self.device = device

        self.lstm = torch.nn.LSTM(self.input_size, self.lstm_size, self.num_layers, batch_first=True, dropout=dropout)
        self.linear1 = torch.nn.Linear(self.lstm_size, self.output_size * 5)
        self.linear2 = torch.nn.Linear(self.output_size*5, self.output_size)

    def forward(self, input):
        #output, self.hidden = self.lstm(input, self.init_weights()) # Useless, same as default torch.LSTM hidden state init.
        output, _ = self.lstm(input)
        output = self.linear1(output)
        output = self.linear2(output)
        return output
        
    # Useless
    #def init_weights(self, batch_size):
        #h0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size)
        #c0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size)
        #h0 = h0.to(self.device)
        #c0 = c0.to(self.device)
        #return Variable(h0), Variable(c0)


class BilinearLSTMSeqNetwork(torch.nn.Module):
    def __init__(self, input_size, out_size, batch_size, device,
                 lstm_size=100, lstm_layers=3, dropout=0.2):
        """
        LSTM network with Bilinear layer
        Input: torch array [batch x frames x input_size]
        Output: torch array [batch x frames x out_size]

        :param input_size: num. channels in input
        :param out_size: num. channels in output
        :param batch_size:
        :param device: torch device
        :param lstm_size: number of LSTM units per layer
        :param lstm_layers: number of LSTM layers
        :param dropout: dropout probability of LSTM (@ref https://pytorch.org/docs/stable/nn.html#lstm)
        """
        super(BilinearLSTMSeqNetwork, self).__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.output_size = out_size
        self.num_layers = lstm_layers
        self.batch_size = batch_size
        self.device = device

        self.bilinear = torch.nn.Bilinear(self.input_size, self.input_size, self.input_size * 4)
        self.lstm = torch.nn.LSTM(self.input_size * 5, self.lstm_size, self.num_layers, batch_first=True, dropout=dropout)
        self.linear1 = torch.nn.Linear(self.lstm_size + self.input_size * 5, self.output_size * 5)
        self.linear2 = torch.nn.Linear(self.output_size * 5, self.output_size)
        self.hidden = self.init_weights()

    def forward(self, input):
        input_mix = self.bilinear(input, input)
        input_mix = torch.cat([input, input_mix], dim=2)
        output, self.hidden = self.lstm(input_mix, self.init_weights())
        output = torch.cat([input_mix, output], dim=2)
        output = self.linear1(output)
        output = self.linear2(output)
        return output

    def init_weights(self):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size)
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        return Variable(h0), Variable(c0)


class TCNSeqNetwork(torch.nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, layer_channels, dropout=0.2):
        """
        Temporal Convolution Network with PReLU activations
        Input: torch array [batch x frames x input_size]
        Output: torch array [batch x frames x out_size]

        :param input_channel: num. channels in input
        :param output_channel: num. channels in output
        :param kernel_size: size of convolution kernel (must be odd)
        :param layer_channels: array specifying num. of channels in each layer
        :param dropout: dropout probability
        """

        super(TCNSeqNetwork, self).__init__()
        self.kernel_size = kernel_size
        self.num_layers = len(layer_channels)

        self.tcn = TemporalConvNet(input_channel, layer_channels, kernel_size, dropout)
        self.output_layer = torch.nn.Conv1d(layer_channels[-1], output_channel, 1)
        self.output_dropout = torch.nn.Dropout(dropout)
        self.net = torch.nn.Sequential(self.tcn, self.output_dropout, self.output_layer)
        self.init_weights()

    def forward(self, x):
        out = x.transpose(1, 2)
        out = self.net(out)
        return out.transpose(1, 2)

    def init_weights(self):
        self.output_layer.weight.data.normal_(0, 0.01)
        self.output_layer.bias.data.normal_(0, 0.001)

    def get_receptive_field(self):
        return 1 + 2 * (self.kernel_size - 1) * (2 ** self.num_layers - 1)