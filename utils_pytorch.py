#!/usr/bin/env python
import torch
from torch import nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0),-1)


class UnFlatten(nn.Module):
    def __init__(self, sizes):
        super(UnFlatten, self).__init__()
        self.sizes = sizes
    def forward(self, input):
        return input.view(input.size(0),*self.sizes)


def fc_block(in_size, out_size, activation='relu', dropout_p=0, batchnorm=True):
    activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()] ])
    layers = [ nn.Linear(in_size, out_size)]
    if dropout_p > 0: layers.append(nn.Dropout(dropout_p))
    if batchnorm: layers.append(nn.BatchNorm1d(out_size))
    layers.append( activations[activation] )
    return nn.Sequential(*layers)

def torch_onehot_to_index(oh_array, axis = 2):
    return torch.argmax(oh_array, axis = axis)

def torch_hamming_dist(array1, array2):
    return torch.sum(torch.ne(array1,array2)) # this sums all the hamming distances together
