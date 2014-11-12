#!/usr/bin/env python
# -*- coding: utf-8 -*-

from array import array
import sys
from os.path import join, abspath

import arrow
import numpy as np
from sklearn.preprocessing import normalize

from interest_scores import InterestScorer
from general import get_matrix_from_to_t
from helpers import repositories_interested_by, users_interested_in
sys.path.append(abspath(join("..", "Recommendation")))
from timed_transition_matrix import TimedTransitionMatrix


def power(P, to):
    P_ = P
    while to > 1:
        P_ = P_ * P
        to -= 1
    return P_

def polysum(P, to):
    if to == 1:
        return P
    else:
        return power(P, to) + polysum(P, to-1)

class MultiMarkovScorer(InterestScorer):
    """Scores according to multiple Markov chains of interests"""
    def __init__(self, interest_metric, dataset_dir, store_atmost):
        super(MultiMarkovScorer, self).__init__(interest_metric, dataset_dir, store_atmost)
        splitted = interest_metric.split("_")
        self.back_weight = float(splitted[1])
        self.k = int(splitted[2])
        transitions_fn = join(dataset_dir, "transitions.npy")
        delta_idxs_fn = join(dataset_dir, "delta_indices.pkl")
        self.ttm = TimedTransitionMatrix(transitions_fn, delta_idxs_fn, self.NR)

    def get_interests_array(self):
        return array('f')

    def get_related_repositories_and_scores(self, uidx, G_I_t, at_t=None):
        N = self.ttm.at_t(at_t)
        N_prime = N + self.back_weight * N.T
        P = normalize(N_prime, norm='l1')
        print "type(P)", type(P)
        sys.exit()
        P_prime = polysum(P, self.k)

        ui_coo = G_I_t[uidx].tocoo()
        if len(ui_coo.data) == 0:
            # logging.debug('Uidx: {} at time {} has no data'.format(uidx, at_t))
            return np.zeros(1), np.zeros(1)
        else:
            prev_ridx = ui_coo.col[np.argmax(ui_coo.data)]

        scores = P_prime[prev_ridx].toarray()[0]
        scores[repositories_interested_by(uidx, G_I_t)] = 0
        rs = scores.nonzero()[0]
        scores = scores[rs]
        return rs, scores
