#!/usr/bin/env python
# -*- coding: utf-8 -*-

from array import array
import cPickle
import os
from os.path import abspath, join, dirname
import sys

from scipy.io import mmread
from scipy.sparse import lil_matrix, coo_matrix
import numpy as np

sys.path.append(abspath(join("..", "Utilities")))
from general import get_matrix_before_t

class TimedTransitionMatrix(object):
    """Base validation transitions matrix + delta"""
    # @profile
    def __init__(self, transitions_fn, delta_indices_fn, num_R):
        self.transitions_fn = abspath(transitions_fn)
        self.delta_indices_fn = abspath(delta_indices_fn)
        self.num_R = num_R

        self.transitions = np.load(self.transitions_fn)
        with open(self.delta_indices_fn, "rb") as pf:
            self.t_to_idx = cPickle.load(pf)

    # @profile
    def at_t(self, t):
        ts_from_to = self.transitions[:,:self.t_to_idx[t]]
        return coo_matrix( (ts_from_to[0],(ts_from_to[1], ts_from_to[2])),
                           shape=(self.num_R, self.num_R) ).tocsr()

# @profile
def get_transitions_array_and_delta_indices(valid_times, all_times, us, rs, num_U):
    i = 0
    before = -np.ones(num_U, dtype=np.int)
    transitions = {}
    idx = 1
    transitions_array = np.zeros((3,1), dtype=np.int)
    t_to_idx = {}

    for t in valid_times:
        while all_times[i] < t:
            uidx = us[i]
            ridx = rs[i]
            prev_ridx = before[uidx]
            if prev_ridx < 0:
                before[uidx] = ridx
                i += 1
                continue

            if (prev_ridx, ridx) in transitions:
                transitions[(prev_ridx, ridx)] += 1
            else:
                transitions[(prev_ridx, ridx)] = 1
            i += 1
            before[uidx] = ridx

        if len(transitions) != 0:
            a = np.array([[c,fr,to] for (fr,to), c in transitions.iteritems()]).T
            idx += len(transitions)
        else:
            a = np.zeros((3,1), dtype=np.int)
            idx += 1

        t_to_idx[t] = idx
        transitions_array = np.concatenate((transitions_array, a), axis=1)
        transitions = {}

    return transitions_array, t_to_idx

if __name__ == '__main__':
    from nose.tools import eq_
    from argparse import ArgumentParser

    argparser = ArgumentParser()
    argparser.add_argument('data_dir', help="directory with the data")
    args = argparser.parse_args()

    data_dir = abspath(args.data_dir)
    G_I = mmread(join(data_dir,"repo_user_times.mtx"))
    num_R = G_I.shape[0]
    num_U = G_I.shape[1]

    with open(join(data_dir,"valid_repos_and_times.pkl"), "rb") as pf:
        valid_times_repos_list = cPickle.load(pf)
    valid_times = np.concatenate([a[0] for a in valid_times_repos_list])
    valid_times.sort()

    ordered_interest_indices = G_I.data.argsort()
    all_times = G_I.data[ordered_interest_indices]
    rs = G_I.row[ordered_interest_indices]
    us = G_I.col[ordered_interest_indices]

    T, d = get_transitions_array_and_delta_indices(valid_times, all_times, us, rs, num_U)

    transitions_fn = join(data_dir, "transitions.npy")
    delta_idxs_fn = join(data_dir, "delta_indices.pkl")
    np.save(transitions_fn, T)
    with open(delta_idxs_fn, "wb") as pf:
        cPickle.dump(d, pf, cPickle.HIGHEST_PROTOCOL)
    print "saved"

    # print "testing"
    # TTM = TimedTransitionMatrix(transitions_fn, delta_idxs_fn, num_R)
    # test_matrix = TTM.at_t(valid_times[1])
    # expected_matrix = coo_matrix(([1,1,1,1,1,1,1,1,1,1,2],([4,10,2,3,8,6,6,7,0,8,1],[1,1,3,6,6,7,8,8,9,10,5])), shape=(num_R, num_R)).tocsr()
    # # print "expected_matrix"
    # # print expected_matrix
    # # print "test_matrix"
    # # print test_matrix
    # # G_I_198000 = get_matrix_before_t(G_I, 198000)
    # # print "G_I_198000"
    # # print G_I_198000
    # eq_( (test_matrix - expected_matrix).nnz, 0 )
    # print "Test pass"

