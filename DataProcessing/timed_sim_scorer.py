#!/usr/bin/env python
# -*- coding: utf-8 -*-

from array import array
import sys

import arrow
import numpy as np

from interest_scores import InterestScorer
from general import get_matrix_from_to_t
from helpers import neighbours, repositories_interested_by
from interest_similarity_scores import get_interest_similarity_scores

class TimedSimilarScorer(InterestScorer):
    """Scores according to similarity of interested users in time interval
       Computes interest scores based on interest similarity
       Adapted from "Amazon.com Recommendations Item-to-Item
       Collaborative Filtering" by Greg Linden, Brent Smith, and Jeremy York
    """
    def __init__(self, interest_metric, dataset_dir, store_atmost):
        super(TimedSimilarScorer, self).__init__(interest_metric, dataset_dir, store_atmost)
        #TAPOP{M,W,D}

        if interest_metric[-1].lower() == 'm':
            self.earlier = {'months':-1}
        elif interest_metric[-1].lower() == 'w':
            self.earlier = {'weeks':-1}
        elif interest_metric[-1].lower() == 'd':
            self.earlier = {'days':-1}
        else:
            print "TimedSimilarScorer Error: 'window'={} is not in ['m', 'w', 'd']".format(interest_metric[-1].lower())
            sys.exit()

    def get_related_repositories_and_scores(self, uidx, G_I_t, at_t=None):
        G_I_interval = get_matrix_from_to_t(G_I_t,
                                            arrow.get(at_t).replace(**self.earlier).timestamp,
                                            at_t).tocsr()
        ns = neighbours(uidx, "ASIM", G_I_t)
        u_rs = repositories_interested_by(uidx, G_I_t)
        rs = np.setdiff1d(repositories_interested_by(ns, G_I_interval),
                          u_rs, assume_unique=True)
        scores = get_interest_similarity_scores(u_rs, rs, G_I_interval)
        return rs, scores
