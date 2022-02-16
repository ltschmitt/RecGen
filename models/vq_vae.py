#!/usr/bin/env python
import torch
import torch.utils.data
from torch import nn
from utils_pytorch import *
from torch.nn import functional as F
import numpy as np
import pandas as pd
from math import prod


class VectorQuantizer(nn.Module):
    """
    Tensorflow original: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    Based on: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings # number of "categories"
        self.D = embedding_dim # number of values representing the "categories"
        self.beta = beta
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K) #? range values seem weird

    def forward(self, latents):
        flat_latents = latents.view(-1, self.D) # from [batch, latent] to [batch*latent/embedding_dim, embedding_dim]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [batch*latent/embedding_dim, num_embeddings]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [batch*latent/embedding_dim, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [batch*latent/embedding_dim, num_embeddings]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [batch, embedding_dim]
        quantized_latents = quantized_latents.view(latents.shape)  # [batch, latent]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        avg_probs = torch.mean(encoding_one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized_latents.contiguous(), vq_loss, perplexity# , encoding_one_hot


class VqEncoder(nn.Module):
    def __init__(self, layer_sizes, **kwargs):
        super(VqEncoder, self).__init__()
        self.flatten = Flatten()
        self.fc_blocks = nn.Sequential(*[fc_block(in_size, out_size, **kwargs) for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:-1])])
        self.fc_z = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc_blocks(x)
        return self.fc_z(x)


class VqDecoder(nn.Module):
    def __init__(self, layer_sizes, output_shape, **kwargs):
        super(VqDecoder, self).__init__()
        self.fc_blocks = nn.Sequential(*[fc_block(in_size, out_size, **kwargs) for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:-1])])
        self.fc_last = nn.Linear(layer_sizes[-2],layer_sizes[-1])
        self.sigmoid = nn.Sigmoid()
        self.unflatten = UnFlatten(output_shape)

    def forward(self, x):
        x = self.fc_blocks(x)
        x = self.fc_last(x)
        x = self.sigmoid(x)
        return self.unflatten(x)


class VQVAE(nn.Module):
    def __init__(self, input_shape, layer_sizes, latent_size, num_embeddings, embedding_dim, layer_kwargs = {}, *args, **kwargs):
        super(VQVAE, self).__init__()
        self.layer_sizes = [prod(input_shape), *layer_sizes, latent_size]
        self.encoder = VqEncoder(self.layer_sizes, **layer_kwargs)
        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim, beta=0.25)
        self.decoder = VqDecoder(self.layer_sizes[::-1], output_shape = input_shape, **layer_kwargs)

    def forward(self, x):
        z = self.encoder(x)
        qz, self.vq_loss, self.perplexity = self.quantizer(z)
        return self.decoder(qz)

    def loss_function(self, recon_x, x, **kwargs):
        #recon_loss = F.mse_loss(recon_x, x, reduction='none')
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='none')
        recon_loss = 0.5*(torch.mean((recon_loss[:,:kwargs.get('ts_len',13)] * kwargs.get('ts_weight',1))) + torch.mean(recon_loss[:,kwargs.get('ts_len',13):]))
        return {'loss': recon_loss + self.vq_loss, 'recon_loss': recon_loss, 'vq_loss': self.vq_loss, 'perplexity': self.perplexity}
