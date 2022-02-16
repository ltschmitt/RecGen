#!/usr/bin/env python
import numpy as np
import pandas as pd
import sys
import os

# onehot conversion functions and lookuptable
vocab_list = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "*"]


def seqaln_to_indices(sequences, vocab_list):
    aa_dict_encode = {c: i for i, c in enumerate(vocab_list)}
    return np.array([[aa_dict_encode[s] for s in seq] for seq in sequences])


def indices_to_seqaln(indices, vocab_list, join = False):
    aa_dict_decode = {i: c for i, c in enumerate(vocab_list)}
    letters = [[aa_dict_decode[i] for i in ind] for ind in indices]
    if join:
        return [''.join(s) for s in letters]
    else:
        return letters


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def onehot_to_index(oh_array, axis = 2):
    return np.argmax(oh_array, axis = axis)


def np_hamming_dist(array1, array2, axis = 1):
    return np.sum(np.not_equal(array1,array2), axis = axis)


def np_hamming_dist_uneven(array1, loop_array, summarise_function = np.mean, axis = 1): # arrays are all (n,357,21)
    if summarise_function == 'random':
        return np_hamming_dist(array1[np.random.randint(0,array1.shape[0],loop_array.shape[0])], loop_array, axis = axis)
    return np.array([summarise_function(np_hamming_dist(s, array1)) for s in loop_array]) # loop through loop_array // compare each loop_array seq with all of array1


def aa_score_ratio(array1, array2, submat):
    return np.sum(submat[array1,array1])/np.sum(submat[array1,array2])


def aa_similarity(array1, array2, submat, axis = 1):
    return array1.shape[0] - np.sum(submat[array1,array2] > 0, axis = axis)


def aa_similarity_uneven(array1, loop_array, summarise_function, submat, axis = 1):
    if summarise_function == 'random':
        return aa_similarity(array1[np.random.randint(0,array1.shape[0],loop_array.shape[0])], loop_array, submat, axis = axis)
    return np.array([summarise_function(aa_similarity(s, array1, submat)) for s in loop_array]) # loop through loop_array // compare each loop_array seq with all of array1


def read_submat(mat_name,vocab_list):
    tab = pd.read_table('ftp://ftp.ncbi.nih.gov/blast/matrices/' + mat_name, comment='#', sep = '\s+')
    return tab.loc[vocab_list,vocab_list].values


def check_mkdir(folderstr, max_iter = 10):
    if os.path.exists(folderstr):
        for i in range(max_iter):
            new_folderstr = folderstr + '_' + str(i)
            if not os.path.exists(new_folderstr):
                break
        if os.path.exists(new_folderstr):
            print('Max repeats reached, aborting!')
            sys.exit()
        print('Folder exists, output will be stored in ' + new_folderstr)
    else: new_folderstr = folderstr
    os.makedirs(new_folderstr, exist_ok=True)
    return new_folderstr
