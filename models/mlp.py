#!/usr/bin/env python
import torch
import torch.utils.data
from torch import nn
from utils_pytorch import *
from torch.nn import functional as F
from math import prod


class Splitter(nn.Module):
    def __init__(self, ts_len, **kwargs):
        super(Splitter, self).__init__()
        self.ts_len = ts_len
        self.flatten = Flatten()

    def forward(self, x):
        y = x[:,:self.ts_len] # target site part
        y_flat = self.flatten(y)
        return y, y_flat


class VaeDecoder(nn.Module):
    def __init__(self, layer_sizes, output_shape, **kwargs):
        super(VaeDecoder, self).__init__()
        self.fc_blocks = nn.Sequential(*[fc_block(in_size, out_size, **kwargs) for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:-1])])
        self.fc_last = nn.Linear(layer_sizes[-2],layer_sizes[-1])
        self.sigmoid = nn.Sigmoid()
        self.unflatten = UnFlatten(output_shape)

    def forward(self, x):
        x = self.fc_blocks(x)
        x = self.fc_last(x)
        x = self.sigmoid(x)
        return self.unflatten(x)


class MLP(nn.Module):
    def __init__(self, input_shape, layer_sizes, ts_len, layer_kwargs={}, *args, **kwargs):
        super(MLP, self).__init__()
        self.splitter = Splitter(ts_len)
        output_shape = (input_shape[0]-ts_len, input_shape[1])
        decoder_layer_sizes = [ts_len*input_shape[1], *layer_sizes[::-1], prod(output_shape)]
        self.decoder = VaeDecoder(decoder_layer_sizes, output_shape = output_shape, **layer_kwargs)

    def forward(self, x):
        y, y_flat = self.splitter(x)
        return torch.cat((y,self.decoder(y_flat)),1)

    def loss_function(self, recon_x, x, **kwargs):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        # change contribution weight of ts to loss - for some weird reason the model does not train with mean readuction

        return {'loss': recon_loss}
