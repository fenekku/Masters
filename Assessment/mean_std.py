#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Return standard deviation of given npy file"""

from argparse import ArgumentParser
import csv

import numpy as np

argparser = ArgumentParser()
argparser.add_argument('array', help="Array")

args = argparser.parse_args()

a = np.load(args.array)

mean = a.mean(axis=0)
std = a.std(axis=0)

print mean, std

with open("unsrdp_results.csv", "ab") as cf:
    w = csv.writer(cf, dialect="excel")
    w.writerow([mean, std])
