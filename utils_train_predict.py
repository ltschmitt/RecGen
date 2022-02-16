#!/usr/bin/env python
import numpy as np
import pandas as pd
import utils
from training import model_predict


def predict_and_index(model, data_input, batch_size):
    yx_pred_oh = model_predict(model, data_input, batch_size)
    yx_pred_ind = utils.onehot_to_index(yx_pred_oh)
    return yx_pred_ind


def hamming_distance_df(yx_pred_ind, yx_ind, ts_len, vocab_list):
    x_hamming = utils.np_hamming_dist(yx_pred_ind[:,ts_len:], yx_ind[:,ts_len:])
    y_hamming = utils.np_hamming_dist(yx_pred_ind[:,:ts_len], yx_ind[:,:ts_len])

    y_str = utils.indices_to_seqaln(yx_ind[:,:ts_len], vocab_list, join = True)

    df = pd.DataFrame({"TargetSequence" : y_str,"Type" : "TargetSite", "HammingDistance" : y_hamming})
    df = pd.concat([df,pd.DataFrame({"TargetSequence" : y_str,"Type" : "Recombinase", "HammingDistance" : x_hamming})])
    return df


def hamming_distance_uneven_df(array1, loop_array, ts_len, ts_labels, vocab_list, summarise_function = np.mean):
    x_hamming = utils.np_hamming_dist_uneven(array1[:,ts_len:], loop_array[:,ts_len:], summarise_function)
    y_hamming = utils.np_hamming_dist_uneven(array1[:,:ts_len], loop_array[:,:ts_len], summarise_function)

    #y_str = utils.indices_to_seqaln(loop_array[:,:ts_len], vocab_list, join = True)
    y_str = utils.indices_to_seqaln(ts_labels, vocab_list, join = True) * x_hamming.shape[0] # ugly fix for missmatch in numbers for df

    df = pd.DataFrame({"TargetSequence" : y_str,"Type" : "TargetSite", "HammingDistance" : y_hamming})
    df = pd.concat([df,pd.DataFrame({"TargetSequence" : y_str,"Type" : "Recombinase", "HammingDistance" : x_hamming})])
    return df


def hamming_similarity_uneven_df(array1, loop_array, ts_len, vocab_list, submat, ts_labels, summarise_function = np.mean):
    x_hamming = utils.np_hamming_dist_uneven(array1[:,ts_len:], loop_array[:,ts_len:], summarise_function)
    y_hamming = utils.np_hamming_dist_uneven(array1[:,:ts_len], loop_array[:,:ts_len], summarise_function)

    x_similarity = utils.aa_similarity_uneven(array1[:,ts_len:], loop_array[:,ts_len:], summarise_function, submat)
    y_similarity = utils.aa_similarity_uneven(array1[:,:ts_len], loop_array[:,:ts_len], summarise_function, submat)

    y_str = utils.indices_to_seqaln(ts_labels, vocab_list, join = True) * x_hamming.shape[0] # ugly fix for missmatch in numbers for df

    df = pd.DataFrame({"TargetSequence" : y_str,"Type" : "TargetSite", "HammingDistance" : y_hamming, "Similarity" : y_similarity})
    df = pd.concat([df,pd.DataFrame({"TargetSequence" : y_str,"Type" : "Recombinase", "HammingDistance" : x_hamming, "Similarity" : x_similarity})])
    return df


def one_aa_freqs(ind_1dim, vocab_list):
    counts = np.bincount(ind_1dim, minlength = len(vocab_list))
    freqs = counts/sum(counts)
    return freqs


def aa_freqs(ind_2dim, vocab_list):
    freqs = [one_aa_freqs(s, vocab_list) for s in ind_2dim.T]
    return pd.DataFrame(freqs, columns = vocab_list)


def make_consensus_ind(sequences,vocab_list):
    ind = utils.seqaln_to_indices(sequences, vocab_list)
    return utils.onehot_to_index(aa_freqs(ind, vocab_list).values, axis = 1) #makes indices consensus sequence


def leave_out_indices(df_series_forfilter, leave_out_y, y_ind, hamming_cutoff = 1):
    testb = df_series_forfilter == leave_out_y
    test_index = list(testb[testb].index)

    # libraries where the TS is equal or above the hamming_cutoff value
    y_hamming = utils.np_hamming_dist(y_ind, np.expand_dims(y_ind[test_index[0]], axis = 0), axis = 1)
    train_index = list(np.flatnonzero(y_hamming >= hamming_cutoff))

    # equal to hamming_cutoff = 1
    #trainb = df_series_forfilter != leave_out_y
    #train_index = list(trainb[trainb].index)

    return train_index, test_index


def z_unif_sampling(z_values, n_samples):
    z_boundaries = [(min(z_values[:,i]), max(z_values[:,i])) for i in range(z_values.shape[1])]
    z_random = np.array([np.random.uniform(min_val,max_val, n_samples) for min_val,max_val in z_boundaries]).T
    return z_random


def z_norm_sampling(z_values, n_samples, mean, spread_deviation = 0.05):
    z_boundaries = [(min(z_values[:,i]), max(z_values[:,i])) for i in range(z_values.shape[1])]
    z_spread = np.mean(np.absolute(np.array(z_boundaries)), axis=1)
    z_random = np.array([np.random.normal(mean_val, sd_val*spread_deviation, n_samples) for mean_val, sd_val in zip(mean,z_spread)]).T
    return z_random


def prediction_ts_distance(x, decoder, ts_len, compare_to_oh): # x is z [?n,latent_size]
    if len(x.shape) == 1:
        x = x.reshape(1,-1)
        sum_axis = (0,1,2)
    else:
        sum_axis = (1,2)

    pred = model_predict(decoder,x)[:,:ts_len] # pred is [n,13,21]
    dist = np.sum(np.absolute(np.subtract(pred,compare_to_oh)), axis = sum_axis) # compare_to_oh should be [1,13,21]
    return dist


def z_search(decoder, z_values, compare_to_oh, ts_len, n_sampling=20000, out_samples=100, loops=2, zoom=0.1):
    z_random = z_unif_sampling(z_values, n_sampling)

    for i in range(loops):
        z_dist = prediction_ts_distance(z_random, decoder, ts_len, compare_to_oh)
        z_mindist_index = np.argmin(z_dist)
        print('dist:' + str(z_dist[z_mindist_index]))
        z_random = z_norm_sampling(z_random, n_sampling, z_random[z_mindist_index], zoom)

    z_dist = prediction_ts_distance(z_random, decoder, ts_len, compare_to_oh)
    z_dist_sort_index = np.argsort(z_dist)
    print('min:' + str(min(z_dist[z_dist_sort_index[:out_samples]])) + ' max:' + str(max(z_dist[z_dist_sort_index[:out_samples]])))
    return z_random[z_dist_sort_index[:out_samples]], z_dist[z_dist_sort_index[:out_samples]] # output the best results, z values and onhot dist 


def z_sample_conditional(decoder, z_values, ts_len, out_samples):
    z_random = z_unif_sampling(z_values, out_samples)
    y_pred_oh = model_predict(decoder, z_random, batch_size = min(10000,n_sampling))[:,:ts_len]
