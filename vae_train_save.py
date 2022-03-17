#!/usr/bin/env python
# Predict a specific sequence
from models import vae_models
import training
import utils
import numpy as np
import pandas as pd
import load_data as ld
import utils_train_predict as utp
import torch
from datetime import datetime
from itertools import chain
import argparse


##### arguments #####

# argument parser
parser = argparse.ArgumentParser(description='Train a VAE with all libraries and save the model file.')
parser.add_argument('-o','--outfolder', nargs='?', default='saved_models/', type=str, help='default = %(default)s; output folder for saving results', dest='outprefix')
parser.add_argument('-i','--input_data', nargs='?', default='example_input/training_data_masked.csv', type=str, help='default = %(default)s; csv input table containing the columns target_sequence and Sequence (recombinase in amino acid).', dest='input_data')
parser.add_argument('-m','--model_type', nargs='?', default='CVAE', type=str, help='default = %(default)s; select the type of VAE model to use; options: VAE, CVAE, SVAE, MMD_VAE, VQ_VAE', dest='model_type')
parser.add_argument('-z','--latent_size', nargs='?', default=2, type=int, help='default = %(default)s; the latent size dimensions', dest='latent_size')
parser.add_argument('-l','--layer_sizes', nargs='*', default=[64,32], type=int, help='default = %(default)s; the hidden layer dimensions in the model', dest='layer_sizes')
parser.add_argument('-b','--batch_size', nargs='?', default=128, type=int, help='default = %(default)s; the number of samples in each processing batch', dest='batch_size')
parser.add_argument('-e','--epochs', nargs='?', default=40, type=int, help='default = %(default)s; the number of iterations the training is going through', dest='epochs')
parser.add_argument('-a','--beta', nargs='?', default=2, type=float, help='default = %(default)s; the final weight on the KL-Divergence', dest='beta')
parser.add_argument('-wd','--weight_decay', nargs='?', default=0, type=int, help='default = %(default)s; a regularisation factor, does not go well with VQ VAE', dest='weight_decay')
parser.add_argument('-tw','--ts_weight', nargs='?', default=1, type=int, help='default = %(default)s; the multiplyer applied to the reconstruction loss of the target site', dest='ts_weight')
parser.add_argument('-r','--nreads', nargs='?', default=10, type=int, help='default = %(default)s; number of protein sequences per library to use for training', dest='nreads')
parser.add_argument('-ts','--ts_slice', nargs='?', default='half', type=str, help='default = %(default)s; what part of the target site to use; options: "half", "+pos14", "full"', dest='ts_slice')
parser.add_argument('--batch_norm', default=False, action='store_true', help='use batch normalisation in the hidden layers', dest='batch_norm')
parser.add_argument('-lr','--learning_rate', nargs='?', default=0.0001, type=float, help='default = %(default)s; the rate of learning, higher means faster learning, but can lead to less accuracy', dest='learning_rate')
parser.add_argument('-d','--dropout_p', nargs='?', default=0.1, type=float, help='default = %(default)s; dropout_p probability for every layer, 0 means no dropout', dest='dropout_p')
parser.add_argument('-D','--num_embeddings', nargs='?', default=10, type=int, help='default = %(default)s; VQ_VAE only, number of "categories" to embed', dest='num_embeddings')
parser.add_argument('-K','--embedding_dim', nargs='?', default=1, type=int, help='default = %(default)s; VQ_VAE only, number of values to represent each embedded "category"', dest='embedding_dim')
parser.add_argument('-n','--n_models', nargs='?', default=1, type=int, help='default = %(default)s; number of models to train', dest='n_models')

args = parser.parse_args()

ts_slice_options = {'half':list(range(13)),'+pos14':list(range(14)) + [20], 'donut':list(range(14)) + list(range(20,34)), 'full':list(range(34)) }
ts_subset_index = ts_slice_options[args.ts_slice]

##### load data #####

combdf = ld.load_Rec_TS(file = args.input_data, nreads = args.nreads, ts_subset_index=ts_subset_index)

# make indices
vocab_list = utils.vocab_list
yx_ind = np.array(utils.seqaln_to_indices(combdf.combined_sequence,vocab_list))
# convert to one hot arrays
yx_oh = utils.get_one_hot(yx_ind, len(vocab_list))

# convert args to dictionary for saving
params = vars(args).copy()
del params['outprefix']
del params['input_data']
del params['model_type']
del params['ts_slice']
params['ts_subset_index'] = ts_subset_index

# where to save and paramter saving
folderstr = args.outprefix
folderstr = utils.check_mkdir(folderstr)
with open(folderstr + "/parameters.txt","w") as f: f.writelines([str(key) + f':\t' + str(val) + '\n' for key, val in params.items()])

###### Train model ########

for i in range(0,args.n_models):
    print('Training model Nr. ' + str(i))
    model = vae_models[args.model_type](
        input_shape=yx_oh.shape[1:],
        layer_sizes=args.layer_sizes,
        latent_size=args.latent_size,
        ts_len=len(ts_subset_index),
        layer_kwargs={'batchnorm':args.batch_norm, 'dropout_p':args.dropout_p})

    model, losses = training.model_training(model, yx_oh, yx_oh, epochs=args.epochs, batch_size=args.batch_size, loss_kwargs={'beta':args.beta}, optimizer_kwargs={'lr':args.learning_rate})

    # save model
    torch.save(model, folderstr + '/' + args.model_type + '_' + str(i) + '.pt')
