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
import re
import os
import argparse


# argument parser
parser = argparse.ArgumentParser(description='Predict recombinases for defined target sites with saved models.')
parser.add_argument('-m','--model_folder', nargs='?', type=str, help='Select the folder where the model files are saved', dest='model_folder')
parser.add_argument('-t','--target_sequences', nargs='?', default='example_input/predict_ts.csv', type=str, help='The target sites you want to predict recombinases for. In csv format, must contain column "target_sequence"', dest='target_sequences')
parser.add_argument('-d','--training_data', nargs='?', default='example_input/training_data_encoded.csv', type=str, help='default = %(default)s; define csv input file with training data. Necessary for estimation of latent space spread.', dest='training_data')
parser.add_argument('-n','--n_out', nargs='?', default=100, type=int, help='default = %(default)s; number of predictions to make for each model', dest='n_out')

args = parser.parse_args()


# define variables
target_sequence = list(pd.read_csv(args.target_sequences).target_sequence)
out_samples = args.n_out
modeldir = args.model_folder

###### load and prepare data ######

# read paramters from models
with open(modeldir + 'parameters.txt') as f:
    params = {k:eval(v) for (k,v) in [x.split(':\t') for x in f.read().splitlines()]}

# get model files from modeldir
regex = re.compile(r'.*.pt$')
model_files = list(filter(regex.search, os.listdir(modeldir)))

# subset target_sequence
target_sequence = [''.join(np.array(list(x))[params['ts_subset_index']]) for x in target_sequence]
ts_len = len(params['ts_subset_index'])

# load training data
combdf = ld.load_Rec_TS(file = args.training_data, nreads = params['nreads'], ts_subset_index=params['ts_subset_index'])

# make indices
vocab_list = utils.vocab_list
yx_ind = np.array(utils.seqaln_to_indices(combdf.combined_sequence,vocab_list))
# convert to one hot arrays
yx_oh = utils.get_one_hot(yx_ind, len(vocab_list))

# convert target sequences to one hot
y_pred_ind = np.array(utils.seqaln_to_indices(target_sequence,vocab_list))
y_pred_oh = utils.get_one_hot(y_pred_ind, len(vocab_list))
ts_oh = np.repeat(np.reshape(a=y_pred_oh, newshape=(len(target_sequence),ts_len*yx_oh.shape[2])), repeats=out_samples, axis=0)

###### Predict ########

pred_str_list = []
for i in model_files:
    print('Predicting with: ' + i)
    model = torch.load(modeldir + i)

    z_train = training.model_predict(model.encoder, yx_oh, 10000)

    z_found = utp.z_unif_sampling(z_values=z_train, n_samples=len(target_sequence)*out_samples)
    z_found = np.concatenate((ts_oh,z_found),1)

    yx_pred_zsearch_ind = utp.predict_and_index(model.decoder, z_found, 0)

    pred_str_list.append(pd.DataFrame({'Sequence' : utils.indices_to_seqaln(yx_pred_zsearch_ind, vocab_list, join = True), 'TargetSequence' : list(chain(*[[x] * out_samples for x in target_sequence])), 'Model' : i }))

# create output folder
folderstr = 'output_prediction/' + datetime.now().strftime("%Y-%m-%d_%H-%M")
folderstr = utils.check_mkdir(folderstr)

# write parameters
params['modeldir'] = modeldir
params['target_sequence'] = target_sequence
with open(folderstr + "/parameters.txt","w") as f: f.writelines([str(key) + f':\t' + str(val) + '\n' for key, val in params.items()])

# write predictions
pd.concat(pred_str_list).to_csv(folderstr + '/prediction_str.csv', index = False)

