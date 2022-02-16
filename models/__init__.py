#!/usr/bin/env python

import models.vae
import models.mmd_vae
import models.vq_vae
import models.conditional_vae
import models.conditional_mmd_vae
import models.mlp

vae_models = {
    'VAE':vae.VAE,
    'CVAE':conditional_vae.CVAE,
    'MMD_VAE':mmd_vae.MMD_VAE,
    'MMD_CVAE':conditional_mmd_vae.MMD_CVAE,
    'VQ_VAE':vq_vae.VQVAE,
    'MLP':mlp.MLP
    }


