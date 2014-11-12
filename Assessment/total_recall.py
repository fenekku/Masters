#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Return total recall"""

from argparse import ArgumentParser
import cPickle
from os.path import abspath

import numpy as np

argparser = ArgumentParser()
argparser.add_argument('recalls_fn', help="Recalls filename")
argparser.add_argument('vtr_fn', help="validating data filename")
args = argparser.parse_args()

#Get necessary data
vtr_fn = abspath(args.vtr_fn)
with open(vtr_fn, "rb") as pf:
    valid_times_repos_list = cPickle.load(pf)

validation_points_per_user = np.array([a.shape[1] for a in valid_times_repos_list])
recalls_per_user = np.load(args.recalls_fn)

true_predictions_per_user = recalls_per_user * validation_points_per_user
true_predictions_per_user = np.rint(true_predictions_per_user).astype(int)
print true_predictions_per_user.sum() / validation_points_per_user.sum(dtype=float)
