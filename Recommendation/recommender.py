#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from os.path import abspath, join, dirname, basename, exists
import shutil
import tempfile
import sys

import numpy as np


class Recommender(object):
    """Abstract Base class for all recommender classes.
       Gathers together common functionality implemented by subunits
       that vary among implementations
    """
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.list_parameters = ["factors", "iterations", "K", "lambda", "neg_d"]

    def __str__(self):
        """Return textual description of actual recommender"""
        return self.__class__.__name__

    def benchmark_fn(self, parameters):
        """Returns the name of the benchmark file
        """
        fn = join(str(self), parameters["metric"])
        for p in self.list_parameters:
            if p in parameters:
                if np.min(parameters[p]) == np.max(parameters[p]):
                    fn += "{}{:.3g}".format(p, parameters[p])
                else:
                    fn += p+"{}-{}".format(np.min(parameters[p]),
                                           np.max(parameters[p]))
        return fn

    @staticmethod
    def load_recommendations(recommendations_fn):
        """Load into memory the list of timed recommendations per user
        """
        archive = np.load(recommendations_fn)
        #Need to sort the archive.files
        sorted_kws = sorted(archive.files, key=lambda s: int(s.split("_")[1]))
        return [archive[kw] for kw in sorted_kws]


    def cleanup(self):
        """remove temporary files"""
        shutil.rmtree(self.tmp_dir)
