#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper functions
"""

from os.path import join, abspath, exists
import gzip
import sys

import numpy as np

sys.path.append(abspath(join("..", "Utilities")))
from general import COHORT_SIZE
from paths import DATA_PATH


def users_interested_in(repo, matrix):
    """Return set of uidx's interested in `repo`
    """
    return np.unique(matrix.tocsc()[:,repo].tocoo().row)

def repositories_interested_by(uidx_or_uidxs, matrix):
    """Return set of repos uidx_or_uidxs is interested in
    """
    return np.unique(matrix[uidx_or_uidxs,:].tocoo().col)

def neighbours(uidx, interest_metric, matrix):
    """Return neighbours of `uidx` in `matrix` according to `interest_metric`:
       for interest_metric that starts with "I" or "A":
           neighbours are the uidxs of users who have shown Interest
           in same repositories as uidx prior to uidx.
       for interest_metric that startswith "S":
            neighbours are social neighbours (1 level apart in the matrix)
    """
    if interest_metric[0] in ["I","A"]:
        m = matrix.astype(np.bool).astype(np.int)
        urs = m[uidx].transpose()
        return np.setdiff1d((m*urs).tocoo().row, [uidx], assume_unique=True)
    elif interest_metric.startswith("S"):
        m = matrix.astype(np.bool).astype(np.int)
        return np.setdiff1d((m[uidx]+m[:,uidx].transpose()).tocoo().col,
                            [uidx], assume_unique=True)
    else:
        return np.array([])

def iter_readme_files(readmes_dir, r_to_i):
    """Returns an iterator that yields the readme file from `readmes_dir`
       for each repository in 'r_to_i'
    """
    for r in xrange(len(r_to_i)):
        rid = r_to_i.r(r)
        readme_fn = join(readmes_dir, str(rid%COHORT_SIZE), str(rid),
                         "README.gz")
        if exists(readme_fn):
            yield gzip.open(readme_fn, "rb")
        else:
            yield open(join(DATA_PATH, "emptyfile"))


if __name__ == "__main__":
    from argparse import ArgumentParser
    import time

    from nose.tools import eq_, ok_
    from scipy.io import mmread

    from paths import VU_TO_I_FN, PROCESSED_DATA_DIR, TIMED_INTERESTS_FN
    from paths import RECOMMENDATION_TIMES_FN, FOLLOWERSHIPS_FN
    from indexer import Indexer
    from general import get_matrix_before_t

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
    u_r_t = mmread(join(dataset_dir, TIMED_INTERESTS_FN)).transpose()
    state = get_matrix_before_t(u_r_t, rt[user]).tocsr()
    nbs = neighbours(user, "APOP", state)
    print "Wall clock time for neighbours: {:.3f} s".format(time.time() - past)

    print "nbs", nbs
    eq_(set(nbs), set([2]))

    nbs = neighbours(user, "IAA", state)
    eq_(set(nbs), set([2]))

    state = mmread(join(dataset_dir, FOLLOWERSHIPS_FN)).tocsr()
    state = get_matrix_before_t(state, rt[user]).tocsr()
    nbs = neighbours(user, "SAA", state)
    eq_(set(nbs), set([2]))


    print "Tests pass"
