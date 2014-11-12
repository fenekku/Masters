#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
from os.path import abspath

from scipy.sparse import coo_matrix


class TimedMatrix(object):
    """Class optimized for getting timed matrices"""
    def __init__(self, timed_matrix_fn):
        self.timed_matrix_fn = abspath(timed_matrix_fn)
        with open(self.timed_matrix_fn, "rb") as pf:
            self.ordered_data = cPickle.load(pf)
            self.ordered_rows = cPickle.load(pf)
            self.ordered_cols = cPickle.load(pf)
            self.shape = cPickle.load(pf)
            self.t_to_idx = cPickle.load(pf)

    # @profile
    def at_t(self, t):
        idx = self.t_to_idx[t]
        return coo_matrix( (self.ordered_data[:idx],(self.ordered_rows[:idx], self.ordered_cols[:idx])),
                           shape=self.shape ).tocsr()

def write_timedmatrix_f(matrix, valid_times, timed_matrix_fn):
    """
        Write pickled file containing TimedMatrix information.

        Parameters
        ----------
        matrix              coo matrix
        valid_times         ndarray (number validation times,)
        timed_matrix_fn     filename to save under
    """
    i = 0
    ordered_indices = matrix.data.argsort()
    ordered_times = matrix.data[ordered_indices]
    ordered_rows = matrix.row[ordered_indices]
    ordered_cols = matrix.col[ordered_indices]
    t_to_idx = {}

    for t in valid_times:
        while ordered_times[i] < t:
            i += 1

        t_to_idx[t] = i

    with open(timed_matrix_fn, "wb") as pf:
        cPickle.dump(ordered_times, pf, -1)
        cPickle.dump(ordered_rows, pf, -1)
        cPickle.dump(ordered_cols, pf, -1)
        cPickle.dump(matrix.shape, pf, -1)
        cPickle.dump(t_to_idx, pf, -1)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from os.path import join, dirname

    from nose.tools import eq_, ok_
    from scipy.io import mmread
    import numpy as np

    from general import get_matrix_before_t


    argparser = ArgumentParser()
    argparser.add_argument('mm_fn', help="Market Matrix filename to convert")
    argparser.add_argument('vrt_fn', help="Valid Repos and Times filename")
    argparser.add_argument('tm_fn', help="Timed Matrix filename")
    args = argparser.parse_args()

    mm_fn = abspath(args.mm_fn)
    vrt_fn = abspath(args.vrt_fn)
    tm_fn = abspath(args.tm_fn)
    data_dir = dirname(args.mm_fn)
    G = mmread(mm_fn)

    with open(vrt_fn, "rb") as pf:
        valid_times_repos_list = cPickle.load(pf)
    valid_times = np.concatenate([a[0] for a in valid_times_repos_list])
    valid_times.sort()

    write_timedmatrix_f(G, valid_times, tm_fn)

    TG = TimedMatrix(tm_fn)

    print "Testing"
    m1 = get_matrix_before_t(G, valid_times[1])
    m2 = TG.at_t(valid_times[1]).tocoo()
    m1.data.sort()
    m2.data.sort()
    ok_(np.all(m1.data == m2.data))
    print "Test pass"
