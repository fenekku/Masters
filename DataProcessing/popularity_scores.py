#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate popularity score matrix
"""

# from array import array
# import os
# from os.path import join, abspath, exists
# import sys
# import time

import numpy as np
# from scipy.io import mmread, mmwrite

# sys.path.append(abspath(join("..", "Utilities")))
# from general import get_matrix_before_t
# from paths import TIMED_INTERESTS_FN, RECOMMENDATION_TIMES_FN

def get_popularity_scores(state, rs=None):
    """Popularity for all repos"""
    if rs is None:
        return state.astype(np.bool).sum(0).A1.astype(np.int16)
    else:
        return state.tocsc()[:,rs].astype(np.bool).sum(0).A1.astype(np.int16)

# class Popularity(object):
#     """Interfaces with the popularity scores stored to disk"""
#     def __init__(self, folder, user_repo_time=None, rt=None):
#         super(Popularity, self).__init__()
#         self.folder = join(folder, "popularity_scores")
#         if not exists(self.folder):
#             print "Generating popularity scores..."
#             os.makedirs(self.folder)

#             if user_repo_time is None:
#                 u_r_t = mmread(join(folder, TIMED_INTERESTS_FN)).transpose()
#             else:
#                 u_r_t = user_repo_time

#             nu, nr = u_r_t.shape

#             if rt is None:
#                 rt = np.load(join(folder, RECOMMENDATION_TIMES_FN))

#             for uidx in xrange(nu):
#                 past = time.time()
#                 state = get_matrix_before_t(u_r_t, rt[uidx]).astype(np.bool).tocsr()
#                 pop_at_t = state.sum(0).A1.astype(np.int16)
#                 np.save(join(self.folder, str(uidx)+".npy"), pop_at_t)
#                 sys.stdout.write("\r{} / {} ({:.3f}s)".format(
#                     uidx+1, nu, time.time()-past))
#                 sys.stdout.flush()

#             print ""

#     def for_user(self, uidx, of=None):
#         popularities = np.load( join(self.folder, str(uidx)+".npy") )
#         if of is not None:
#             return popularities[of]
#         return popularities


if __name__ == "__main__":
    from argparse import ArgumentParser
    import time

    from nose.tools import eq_, ok_
    from paths import VU_TO_I_FN, PROCESSED_DATA_DIR
    from indexer import Indexer

    argparser = ArgumentParser()
    argparser.add_argument('version', help="dataset version")
    args = argparser.parse_args()

    dataset_dir = join(PROCESSED_DATA_DIR, "test", args.version)
    rt = np.load(join(dataset_dir, RECOMMENDATION_TIMES_FN))

    u_to_i = Indexer(join(dataset_dir, VU_TO_I_FN))
    real_user = 1
    user = u_to_i[real_user]
    print "{} (real)-> {} (idx)-> {} (mm idx)".format(real_user, user, user+1)
    print "rt[{}] = {}".format(user, rt[user])
    past = time.time()
    p = Popularity(dataset_dir)
    print "Wall clock time for POP: {:.3f} s".format(time.time() - past)
    p_user = p.for_user(user)
    print "p_user", p_user
    eq_(set(p_user), set([1,1,1,1,2,1,1,2,0,0,2]))
    print "Tests pass"
