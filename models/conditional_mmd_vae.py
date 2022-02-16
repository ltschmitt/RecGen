#!/usr/bin/env python
import torch
import torch.utils.data
from torch import nn
from utils_pytorch import *
from torch.nn import functional as F
from math import prod


class VaeEncoder(nn.Module):
    def __init__(self, layer_sizes, ts_len, **kwargs):
        super(VaeEncoder, self).__init__()
        self.ts_len = ts_len
        self.flatten = Flatten()
        self.fc_blocks = nn.Sequential(*[fc_block(in_size, out_size, **kwargs) for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:-1])])
        self.fc_mu = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    def forward(self, x):
        y = x[:,:self.ts_len] # target site part
        #x = x[:,self.ts_len:] # protein part - original is without it
        x = self.flatten(x)
        x = self.fc_blocks(x)
        return self.fc_mu(x), y


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


class MMD_CVAE(nn.Module):
    def __init__(self, input_shape, layer_sizes, latent_size, ts_len, layer_kwargs={}, *args, **kwargs):
        super(MMD_CVAE, self).__init__()
        self.input_shape = (input_shape[0]-ts_len, input_shape[1])
        self.layer_sizes = [prod(input_shape), *layer_sizes, latent_size]
        self.encoder = VaeEncoder(self.layer_sizes, ts_len, **layer_kwargs)
        self.dec_layer_sizes = [latent_size+ts_len*input_shape[1], *layer_sizes[::-1], prod(self.input_shape)]
        self.decoder = VaeDecoder(self.dec_layer_sizes, output_shape = self.input_shape, **layer_kwargs)

    def forward(self, x):
        self.z, y = self.encoder(x)
        z = torch.cat((y.view(-1, prod(y.shape[1:])), self.z), 1) # combine ts with z
        return torch.cat((y,self.decoder(z)),1)

    def gaussian_kernel(self, a, b):
        dim1_1, dim1_2 = a.shape[0], b.shape[0]
        depth = a.shape[1]
        a = a.view(dim1_1, 1, depth)
        b = b.view(1, dim1_2, depth)
        a_core = a.expand(dim1_1, dim1_2, depth)
        b_core = b.expand(dim1_1, dim1_2, depth)
        numerator = (a_core - b_core).pow(2).mean(2)/depth
        return torch.exp(-numerator)

    def compute_mmd(self, a, b):
        return self.gaussian_kernel(a, a).mean() + self.gaussian_kernel(b, b).mean() - 2*self.gaussian_kernel(a, b).mean()

    def loss_function(self, recon_x, x, **kwargs):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='none')
        recon_loss = 0.5*(torch.mean((recon_loss[:,:kwargs.get('ts_len',13)] * kwargs.get('ts_weight',1))) + torch.mean(recon_loss[:,kwargs.get('ts_len',13):]))

        reference_distribution = torch.randn(1000, self.z.shape[1], requires_grad=False)
        reference_distribution = reference_distribution.to(torch.device('cuda'))
        mmd_loss = self.compute_mmd(reference_distribution, self.z)
        adj_mmd = kwargs.get('beta',1) * mmd_loss

        return {'loss': recon_loss + adj_mmd, 'recon_loss': recon_loss, 'mmd_loss': mmd_loss, 'adj_mmd': adj_mmd}
