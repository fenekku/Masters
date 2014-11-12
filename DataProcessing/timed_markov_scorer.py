#!/usr/bin/env python
# -*- coding: utf-8 -*-

from array import array
import sys

import arrow
import numpy as np

from interest_scores import InterestScorer
from general import get_matrix_from_to_t
from helpers import repositories_interested_by, users_interested_in

class TimedMarkovScorer(InterestScorer):
    """Scores according to Markov chains of interests in time interval"""
    def __init__(self, interest_metric, dataset_dir, store_atmost):
        super(TimedMarkovScorer, self).__init__(interest_metric, dataset_dir, store_atmost)
        #TMkvX.X{M,W,D}
        self.back_weight = float(interest_metric[4:7])

        if interest_metric[-1].lower() == 'm':
            self.earlier = {'months':-1}
        elif interest_metric[-1].lower() == 'w':
            self.earlier = {'weeks':-1}
        elif interest_metric[-1].lower() == 'd':
            self.earlier = {'days':-1}
        else:
            print "TimedMarkovScorer Error: 'window'={} is not in ['m', 'w', 'd']".format(interest_metric[-1].lower())
            sys.exit()

    def get_interests_array(self):
        return array('f')

    def get_related_repositories_and_scores(self, uidx, G_I_t, at_t=None):
        ui_coo = G_I_t[uidx].tocoo()
        scores = np.zeros(self.NR)

        if len(ui_coo.data) == 0:
            # logging.debug('Uidx: {} at time {} has no data'.format(uidx, at_t))
            return np.zeros(1), np.zeros(1)
        else:
            prev_ridx = ui_coo.col[np.argmax(ui_coo.data)]

        G_I_interval = get_matrix_from_to_t(G_I_t,
                                            arrow.get(at_t).replace(**self.earlier).timestamp,
                                            at_t).tocsr()

        related_users = users_interested_in(prev_ridx, G_I_interval)

        for user in related_users:
            G_I_t_u_coo = G_I_interval[user].tocoo()
            ordered_args = np.argsort(G_I_t_u_coo.data)
            ordered_interests = G_I_t_u_coo.col[ordered_args]

            arg_prev_ridx = (ordered_interests == prev_ridx).nonzero()[0][0]

            if arg_prev_ridx-1 >= 0:
                prev_prev_ridx = ordered_interests[arg_prev_ridx-1]
                scores[prev_prev_ridx] += self.back_weight #previous less important than next

            if arg_prev_ridx+1 < len(ordered_interests):
                next_prev_ridx = ordered_interests[arg_prev_ridx+1]
                scores[next_prev_ridx] += 1 #next more important

        scores[repositories_interested_by(uidx, G_I_t)] = 0
        rs = scores.nonzero()[0]
        scores = scores[rs]
        return rs, scores