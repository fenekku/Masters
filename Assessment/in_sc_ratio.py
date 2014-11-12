#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""return % of total recommendations that belong to a social component"""

from argparse import ArgumentParser
import cPickle
from os.path import abspath

import numpy as np

argparser = ArgumentParser()
argparser.add_argument('in_sc_fn', help="in sc filename")
argparser.add_argument('vtr_fn', help="validating data filename")
argparser.add_argument('K', type=int, help="K")
args = argparser.parse_args()

#Get necessary data
recommendations_in_sc_per_user = np.load(args.in_sc_fn)
vtr_fn = abspath(args.vtr_fn)
with open(vtr_fn, "rb") as pf:
    validation_tr_per_user = cPickle.load(pf)
validation_points_per_user = np.array([a.shape[1] for a in validation_tr_per_user])

recommendations_per_user = validation_points_per_user * args.K

overall_ratio_insc = recommendations_in_sc_per_user.sum() / recommendations_per_user.sum(dtype=float)
ratio_insc_per_user = recommendations_in_sc_per_user / recommendations_per_user.astype(np.float)
print "(Total percentage, Average) of recommendations in social component"
print (overall_ratio_insc, np.mean(ratio_insc_per_user))

# true_predictions_per_user = recalls_per_user * validation_points_per_user
# true_predictions_per_user = np.rint(true_predictions_per_user).astype(int)
# print true_predictions_per_user.sum() / validation_points_per_user.sum(dtype=float)
