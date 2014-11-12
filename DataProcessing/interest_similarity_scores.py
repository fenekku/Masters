#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate similarity interest score matrix
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from helpers import repositories_interested_by


def get_interest_similarity_scores(u_rs, nb_rs, state, similarity="cos"):
    """Return ndarray of the maximum `similarity` scores of each `nb_rs` wrt
       the repositories `uidx` is interested in at `state`.
    """
    if len(u_rs) == 0:
        #training time equal to validation time -> rare as it implies
        #several actions occurred at the same second
        #ignore them for now (only 8 of them)
        return np.array([])

    r_u_csr = state.transpose().astype(np.bool).astype(np.int).tocsr()

    if similarity == "cos":
        cos_sim = cosine_similarity(r_u_csr[nb_rs,:], r_u_csr[u_rs,:])
        max_similarity_scores = cos_sim.max(axis=1)

    return max_similarity_scores


if __name__ == "__main__":
    pass
    # from argparse import ArgumentParser
    # import time
    # import sys

    # from nose.tools import eq_, ok_

    # sys.path.append(abspath(join("..", "Utilities")))
    # from paths import VU_TO_I_FN, PROCESSED_DATA_DIR, RECOMMENDATION_TIMES_FN
    # from paths import TIMED_INTERESTS_FN, VR_TO_I_FN
    # from indexer import Indexer
    # from general import get_matrix_before_t


    # argparser = ArgumentParser()
    # argparser.add_argument('version', help="dataset version")
    # args = argparser.parse_args()

    # dataset_dir = join(PROCESSED_DATA_DIR, "test", args.version)

    # u_to_i = Indexer(join(dataset_dir, VU_TO_I_FN))
    # print "u_to_i", u_to_i
    # r_to_i = Indexer(join(dataset_dir, VR_TO_I_FN))
    # print "r_to_i", r_to_i
    # real_user = 1
    # uidx = u_to_i[real_user]
    # print "{} (real)-> {} (idx)-> {} (mm idx)".format(real_user, uidx, uidx+1)

    # past = time.time()

    # rt = np.load(join(dataset_dir,RECOMMENDATION_TIMES_FN))
    # print "rt[{}] = {}".format(uidx,rt[uidx])
    # state = get_matrix_before_t(mmread(join(dataset_dir,TIMED_INTERESTS_FN)).transpose(),
    #                             rt[uidx]).tocsr()
    # s = get_interest_similarity_scores(uidx, state, dataset_dir, similarity="cos")
    # print "sim for uidx {}".format(uidx)
    # print s
    # eq_(s.nnz, 8)
    # print "Wall clock time for ASIM: {:.3f} s".format(time.time() - past)
    # print "Tests pass"
