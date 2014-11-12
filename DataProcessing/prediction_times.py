#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pickle a list of user prediction times (times at which they show a
new external interest)
"""

import cPickle
from os.path import join, abspath
import sys

# import arrow
import numpy as np
from scipy.io import mmread

sys.path.append(abspath(join("..", "Utilities")))
from paths import VALIDATING_FN, TIMED_INTERESTS_FN
from paths import VALID_REPOS_AND_TIMES
from general import timedfunction


@timedfunction
def generate_valid_repos_and_times(dataset_dir):
    """Function called to generate VALID_REPOS_AND_TIMES in `dataset_dir`
    """
    valid_repos_and_times = []

    repos_users_times_fn = join(dataset_dir, TIMED_INTERESTS_FN)
    u_r_t = mmread(repos_users_times_fn).transpose().tocsr()

    validation_repos_fn = join(dataset_dir, VALIDATING_FN)
    validation_matrix = mmread(validation_repos_fn).tocsr()

    v_u_r_t = u_r_t.multiply(validation_matrix).tolil()

    for uidx in xrange(v_u_r_t.shape[0]):
        v_r_t_coo = v_u_r_t.getrowview(uidx).tocoo()
        sorted_index = np.argsort(v_r_t_coo.data)

        times = v_r_t_coo.data[sorted_index]
        repos = v_r_t_coo.col[sorted_index]
        valid_repos_and_times.append(np.vstack((times,repos)))

    pt_fn = join(dataset_dir, VALID_REPOS_AND_TIMES)
    with open(pt_fn, "wb") as pf:
        cPickle.dump(valid_repos_and_times, pf, cPickle.HIGHEST_PROTOCOL)
    return pt_fn


if __name__ == '__main__':
    from argparse import ArgumentParser

    from nose.tools import eq_, ok_

    from paths import VU_TO_I_FN, VR_TO_I_FN, PROCESSED_DATA_DIR
    from indexer import Indexer


    argparser = ArgumentParser()
    argparser.add_argument('version', help="dataset version")
    argparser.add_argument('--test', help="Is this a test or not",
                           action="store_true")
    args = argparser.parse_args()

    if args.test:
        dataset_dir = join(PROCESSED_DATA_DIR, "test", args.version)
    else:
        dataset_dir = join(PROCESSED_DATA_DIR, args.version)

    pt_fn = generate_valid_repos_and_times(dataset_dir)

    if args.test:
        with open(pt_fn, "rb") as pf:
            pt = cPickle.load(pf)

        u_to_i = Indexer(join(dataset_dir, VU_TO_I_FN))
        r_to_i = Indexer(join(dataset_dir, VR_TO_I_FN))
        user = 1
        uidx = u_to_i[user]
        eq_(pt[uidx].shape, (2,2))
        print "pt[uidx]", pt[uidx]
        ept = np.array([[160000, 200000],[r_to_i[5], r_to_i[10]]])
        print "expected_pt[uidx]", ept
        ok_( np.all(pt[uidx] == ept) )

        print "Tests pass"
