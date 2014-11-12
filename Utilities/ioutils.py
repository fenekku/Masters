#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Input/output functions used all over
"""

import os
from os.path import join, splitext, exists, dirname
from datetime import datetime
from itertools import izip
import calendar

import numpy as np
from scipy.sparse import coo_matrix


def to_file(iter_cursor, filename):
    """
    Write the results of the iter_cursor to a file with filename
    in the wel format
    """

    def _reformat(r):
        conv = lambda x: str(to_timestamp(x)) if type(x) == datetime else str(x)
        return tuple(map(conv, r))

    with open(filename, 'w') as out_file:
        for r in iter_cursor:
            out_file.write("\t".join(_reformat(r)) + "\n")


def save_sparse(name, sparse_matrix):
    """Saves sparse_matrix to disk"""
    dirs = dirname(name)
    if dirs and not exists(dirs):
        os.makedirs(dirs)

    sparse_matrix_csr = sparse_matrix.tocsr()
    np.savez(name,
             data = sparse_matrix_csr.data,
             indices = sparse_matrix_csr.indices,
             indptr = sparse_matrix_csr.indptr,
             shape = np.asarray(sparse_matrix_csr.shape))

def load_sparse(name):
    """loads sparse_matrix saved to disk via save_sparse"""
    from scipy.sparse import csr_matrix
    ext = "" if splitext(name)[1] == ".npz" else ".npz"
    a = np.load(name+ext)
    return csr_matrix( (a["data"], a["indices"], a["indptr"]),
                       shape=a["shape"] )


def mmm_iter(modifiedmatrixmarket_fn):
    """
    Yield the content of the modified matrixmarket file one converted
    line at a time.
    """
    with open(modifiedmatrixmarket_fn) as f:
        seen_header = False

        for line in f:
            if line.startswith('%'):
                continue

            row = [int(x)-1 for x in line.split()]

            if len(row) == 2 and not seen_header:
                seen_header = True
                rows, cols = row
            else:
                yield row

def mmmread(modifiedmatrixmarket_fn):
    """
    Return a 2-D np.array corresponding to the modified matrixmarket
    format
    """
    m = np.loadtxt(modifiedmatrixmarket_fn, comments="%", skiprows=3)
    if modifiedmatrixmarket_fn.endswith(".ids"):
        return m[:,1:].astype(np.int) - 1
    else:
        return m[:,1:]


def mmmwrite(modifiedmatrixmarket_fn, matrix):
    """
    Write `matrix` to the modified matrixmarket format.
    """
    with open(modifiedmatrixmarket_fn, 'w') as f:
        f.write("%%MatrixMarket matrix array real general\n")
        f.write("% This format does not correspond to an actual")
        f.write(" MatrixMarket but it is close.\n")
        f.write("% The next line designates the rows ")
        f.write("and cols and the subsequent lines correspond " \
                "to the matrix actual content.\n")
        f.write(" ".join(map(str,matrix.shape)) + "\n")
        for row in matrix:
            f.write(" ".join(map(str,row))+"\n")


def readratings(ids_fn):
    """Return top K 2-D array according to ids_fn
    """
    ids_matrix = np.loadtxt(ids_fn, comments="%", skiprows=3) - 1
    return ids_matrix[:,1:]

def bmread(benchmark_fn):
    """Read a .benchmark file
       Takes
        `benchmark_fn`  file name
       Returns a benchmark dict with the following entries:
         metric_str - str : name of the metric used
         fixed_parameters - dict : non-varying parameters
         varying_parameters - tuple : ('name', values)
         avgs - dict : ndarray
         stddevs - dict : ndarray
         fn - the filename
    """
    benchmark = {}
    benchmark["fn"] = benchmark_fn
    with open(benchmark_fn) as bf:
        #Metric
        benchmark["metric"] = bf.readline().strip()
        #Fixed parameters
        fixed_parameters = bf.readline().strip().split()
        benchmark["fixed_parameters"] = {}
        if len(fixed_parameters) % 2:
            raise Exception("bmread error: uneven number of fixed parameter value pairs")

        for i in xrange(0, len(fixed_parameters), 2):
            #keep them as strings
            try:
                value = int(fixed_parameters[i+1])
            except ValueError:
                value = float(fixed_parameters[i+1])
            benchmark["fixed_parameters"][fixed_parameters[i]] = value

        #Varying parameters
        items = bf.readline().strip().split()
        try:
            values = np.array(map(int, items[1:]))
        except ValueError:
            values = np.array(map(float, items[1:]))
        benchmark["varying_parameter"] = (items[0], values)

        #Data
        benchmark["avgs"] = np.fromstring(bf.readline(), sep=" ")
        benchmark["stddevs"] = np.fromstring(bf.readline(), sep=" ")

    return benchmark

def bmwrite(fn, metric_str, fixed_parameters, varying_parameter,
            avgs, stddevs):
    """Write a .benchmark file
        `fn`  file name
        `metric_str`  considered metric name
        `fixed_parameters`  dictionary of fixed 'name' and value
        `varying_parameter`  ('name', values) tuple - values are for the x-axis
        `avgs`  actual 1-D array of data
        `stddevs`  corresponding 1-D array of std_dev
    """
    with open(fn, "w") as bf:
        #Metric
        bf.write(metric_str+"\n")
        #Fixed parameters
        for pname, pvalue in fixed_parameters.iteritems():
            if type(pvalue) != str:
                bf.write("%s %g " % (pname, pvalue))
        bf.write("\n")
        #Varying parameters
        bf.write(varying_parameter[0]+" "+
            " ".join("%g" % pvalue for pvalue in varying_parameter[1]))
        bf.write("\n")

        #avgs
        np.savetxt(bf, avgs, newline=" ", fmt='%g')
        bf.write("\n")
        #stddevs
        np.savetxt(bf, stddevs, newline=" ", fmt='%g')


if __name__ == '__main__':
    from nose.tools import eq_, ok_

    def _equal(a,b):
        """
        compares two sparse matrices
        """
        acoo = coo_matrix(a)
        bcoo = coo_matrix(b)

        eq_(acoo.shape, bcoo.shape)
        eq_(len(acoo.row), len(bcoo.row))
        eq_(len(acoo.col), len(bcoo.col))
        eq_(len(acoo.data), len(bcoo.data))

        for ar,ac,ad, br,bc,bd in zip(acoo.row, acoo.col, acoo.data,
                                      bcoo.row, bcoo.col, bcoo.data):
            eq_(ar,br)
            eq_(ac,bc)
            eq_(ad,bd)

    # a = lil_matrix( (4,4), dtype=bool)
    # a[0,1] = True
    # a[1,2] = True
    # a[1,1] = True
    # save_sparse("testa", a)
    # b = load_sparse("testa")
    # _equal(a, b.tolil())

    # l = [True,False,True]
    # f = argfind(l)
    # eq_(f,0)
    # f = argfind(l, reverse=True)
    # eq_(f,2)

    # V = mmmread("testbV.mm")
    # U = mmmread("testbU.mm")

    # v = np.array([[5.22e-01, 8.19e-01, 8.79e-01],
    #               [3.59e-01, 8.64e-01, 2.86e-01]])
    # u = np.array([[3.53e-01, 1.14e-02, 5.40e-01],
    #               [4.68e-02, 2.52e-01, 4.78e-01],
    #               [3.13e-01, 6.25e-01, 4.82e-01],
    #               [3.40e-01, 5.13e-01, 1.89e-01],
    #               [9.13e-01, 2.44e-01, 3.17e-01],
    #               [6.89e-01, 8.97e-01, 5.72e-01],
    #               [4.48e-01, 1.36e-01, 6.75e-01]])
    # _equal(u, U)
    # _equal(v, V)

    # mmmwrite("testbV.mm", V)
    # b = mmmread("testbV.mm")
    # _equal(V, b)

    # R_ = readrankings("test.ids", "test.ratings", (2,4))
    # V_ = coo_matrix((v.flatten(), ([0,0,0,1,1,1], [0,1,2,0,1,2])),
    #                 shape=(2,4) )
    # _equal(R_, V_)

    # topV = topK(V_, 2)
    # topv = np.array([[2, 1],[1,0]])
    # _equal(topV, topv)

    fixed_params = {"runs":10}
    vp = ("K",[10,20,40])
    metric = "recall"

    #One varying parameter
    data = np.arange(3)
    stddev = np.array([10**(-x) for x in xrange(3)])
    bmwrite("test1", metric, fixed_params, vp, data, stddev)
    bmf = bmread("test1.benchmark")
    eq_(bmf["metric"], "recall")
    eq_(bmf["fixed_parameters"], fixed_params)
    eq_(bmf["varying_parameter"][0], vp[0])
    ok_(np.all(bmf["varying_parameter"][1] == vp[1]))
    ok_(np.all(bmf["avgs"] == data))
    ok_(np.all(bmf["stddevs"] == stddev))

    print "Tests pass"

