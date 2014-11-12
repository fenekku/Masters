#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions used all over
"""

import os
from os.path import join, splitext, exists, dirname
from itertools import izip
import time

import numpy as np
from scipy.sparse import coo_matrix

import psycopg2


def getDB(is_test=False):
    """
    Return the database according to where the script is:
        - on my computer
        - on slipstream
    """
    if os.environ["USER"] == "guillaume":
        if is_test:
            databaseName = "minighdb"
        else:
            databaseName = "github_activities3"
    else: #On slipstream
        databaseName = "github_activities_db2"
    return psycopg2.connect(database = databaseName,
                            user = os.environ["USER"])

def _div(a, by):
    """Sane division"""
    return a if by == 0 else a / by

div = np.vectorize(_div)


def abs_matrix(m):
    """Return matrix with absolute value"""
    a = m.tocsr()
    for i in xrange(len(a.data)):
        a.data[i] = -a.data[i] if a.data[i] < 0 else a.data[i]
    return a


def print_distribution(array):
    if len(array) == 0:
        print "Distribution is empty"
        return
    print "Number of points in the distribution: {}".format(len(array))
    print "\tValues range : [{}, {}]".format(np.min(array),
                                             np.max(array))
    print "\t10th percentile:", np.percentile(array, 10)
    print "\t25th percentile:", np.percentile(array, 25)
    print "\tmedian:", np.percentile(array, 50)
    print "\tmean:", np.mean(array)
    print "\t75th percentile:", np.percentile(array, 75)
    print "\t90th percentile:", np.percentile(array, 90)
    print "\tSum of point values:", np.sum(array)
    print ""


def argfind(l, condition=lambda e: e, reverse=False):
    """Returns the index of the first element from `l`
       that matches `condition`.
       `condition` takes an element as input
       `reverse` can be used to start the search from the end,
       effectively returning the index of the last element from
       `l` matching `condition`.
       This should be in the standard library but is not.
       Returns -1 if no element matches `condition`
    """
    if reverse:
        indices = xrange(len(l)-1,-1,-1)
    else:
        indices = xrange(len(l))

    for i in indices:
        if condition(l[i]):
            return i
    return -1


def confusion_matrix(true_values, predicted_values):
    """Return a dictionary representing a binary confusion matrix"""
    true_values_ = true_values.tocoo()
    confusionmatrix = {"true_positives":0, "false_positives":0,
                        "false_negatives":0, "true_negatives":0}

    for r, c in izip(true_values_.row, true_values_.col):
        if true_values[r,c] == 1:
            if true_values[r,c] == predicted_values[r,c]:
                confusionmatrix["true_positives"] += 1
            else:
                confusionmatrix["false_negatives"] += 1
        else:
            if true_values[r,c] == predicted_values[r,c]:
                confusionmatrix["true_negatives"] += 1
            else:
                confusionmatrix["false_positives"] += 1

    return confusionmatrix


def topK(ratings, K):
    """Returns 2-D numpy array containing topK ridxs for each user"""
    r = ratings.tocsr()
    topks = np.zeros( (r.shape[0], K), dtype=np.int )
    for uidx, ratings in enumerate(r):
        ratings = ratings.toarray()[0]
        topks[uidx] = np.argsort(ratings)[-K:][::-1]
    return topks


def get_negative_density(sparse_matrix):
    """Return negative density of a sparse matrix"""
    m_csr = sparse_matrix.tocsr()
    m_coo = sparse_matrix.tocoo()
    denominator = np.multiply(*m_csr.shape)
    numerator = 0.0

    for u,r in izip(m_coo.row, m_coo.col):
        if m_csr[u,r] < 0:
            numerator += 1

    return numerator / denominator


def timedmethod(method):
    """Decorator used to easily output running time of a method"""
    def _timed(*args, **kwargs):
        wall_past = time.time()
        result = method(*args,**kwargs)
        wall_present = time.time()
        print "Wall clock time for %s.%s(): %.3f s" % (args[0].__class__.__name__,
                                                       method.__name__,
                                                       wall_present - wall_past)
        return result
    return _timed

def timedfunction(function):
    """Decorator used to easily output running time of a function"""
    def _timed(*args, **kwargs):
        wall_past = time.time()
        result = function(*args,**kwargs)
        wall_present = time.time()
        print "Wall clock time for %s(): %.3f s" % (function.__name__,
                                                    wall_present - wall_past)
        return result
    return _timed

def get_matrix_before_t(matrix, t):
    m = matrix.tocoo()
    mask = m.data < t
    return coo_matrix((m.data[mask], (m.row[mask], m.col[mask])),
                       shape=m.shape)

def get_matrix_from_to_t(matrix, t0, t1):
    m = matrix.tocoo()
    mask = (t0 <= m.data) & (m.data < t1)
    return coo_matrix((m.data[mask], (m.row[mask], m.col[mask])),
                       shape=m.shape)


import numpy.core.numeric as _nx

def nans_to_num(x, num, inplace=False):
    """
        Replace nan with zero and inf with a finite number.
        Returns an array replacing Not a Number (NaN) with zero,
        (positive) infinity with `num`.

        Parameters
        ----------
        x : array_like
        num : number to replace inf by
        inplace : modify x

        Input data.
        Returns
        -------
        out : ndarray, float
        Array with the same shape as `x` and dtype of the element in `x` with
        the greatest precision. NaN is replaced by zero, and infinity
        is replaced by num. All finite numbers
        are upcast to the output dtype (default float64).
        Adapted from numpy.nan_to_num
    """
    try:
        t = x.dtype.type
    except AttributeError as ae:
        t = obj2sctype(type(x))

    if inplace:
        y = x
    else:
        try:
            y = x.copy()
        except AttributeError:
            y = array(x)

    if not issubclass(t, _nx.integer):
        if not y.shape:
            y = array([x])
            scalar = True
        else:
            scalar = False

        are_inf = np.isposinf(y)
        are_nan = np.isnan(y)
        y[are_nan] = 0
        y[are_inf] = num

        if scalar:
            y = y[0]

    return y

def imin(x):
    """improved min which returns 0 if x has no elements
    """
    if x.size == 0:
        return 0
    else:
        return np.min(x)

#Global enums
FOLLOWING = 0
CONTRIBUTING = 1
WATCHING = 2

COHORT_SIZE = 5000


if __name__ == '__main__':
    from scipy.sparse import lil_matrix, csgraph
    from nose.tools import eq_

    a = lil_matrix((6,6), dtype=np.int)
    a[0,1] = 2
    a[0,2] = 1
    a[1,2] = 1
    a[2,3] = 1
    a[4,5] = 1
    d = csgraph.shortest_path(a, directed=False, unweighted=True)
    eq_(d[0,0], 0)
    eq_(d[0,1], 1)
    eq_(d[1,0], 1)
    eq_(d[0,3], 2)
    eq_(d[0,5], np.inf)

    a[0,1] = -2
    a[2,3] = -1
    eq_(get_negative_density(a), 1/18.0)

    print "Tests pass"

