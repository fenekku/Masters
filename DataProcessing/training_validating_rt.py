#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build sparse matrices and recommendation times array:
    training
        2/3 external interests +
        all internal before first external of remaining 1/3
    validating
        remaining 1/3 external interests
    recommendation_times
        array of times of recommendation

                    number repos
    number user | BoolInterest_{uidx,r}
"""

from os.path import join, abspath
import sys
from math import floor

import numpy as np
from scipy.sparse import lil_matrix
from scipy.io import mmwrite, mmread

sys.path.append(abspath(join("..", "Utilities")))
from general import getDB
from paths import PROCESSED_DATA_DIR, TRAINING_FN, VALIDATING_FN
from paths import RECOMMENDATION_TIMES_FN
from general import timedfunction


@timedfunction
def generate_training_validating_rt(version, r_to_i, u_to_i, r_u_t_fn,
                                    split, is_test=False):
    """Function called to generate training.mtx, validating.mtx and
       recommendation_times.npy
    """
    if is_test:
        data_processed_dir = join(PROCESSED_DATA_DIR, "test")
    else:
        data_processed_dir = PROCESSED_DATA_DIR

    u_r_times = mmread(r_u_t_fn).transpose().tolil()
    nu, nr = u_r_times.shape

    training_matrix = lil_matrix((nu,nr), dtype=np.int_)
    validating_matrix = lil_matrix((nu,nr), dtype=np.int_)
    recommendation_times = np.zeros(nu, dtype=np.int_)

    valid_repositories_table = version+"_repositories"
    cursor = getDB(is_test=is_test).cursor()

    for uidx in xrange(nu):
        cursor.execute("""SELECT vr.id
                          FROM repositories as r,
                               {} as vr
                          WHERE vr.id = r.id AND r.owner_id = %s
                       """.format(valid_repositories_table), (u_to_i.r(uidx),))
        owned_rs = np.array([r_to_i[r[0]] for r in cursor])
        interests = u_r_times.getrowview(uidx)
        interested_rs = np.unique(interests.tocoo().col)
        ext_rs = np.setdiff1d(interested_rs, owned_rs, assume_unique=True)
        times = interests[0,ext_rs].toarray()[0]
        sorted_indices = times.argsort()
        threshold = int(floor(split*len(ext_rs)))
        training = [ext_rs[i] for i in sorted_indices[:threshold]]
        threshold_time = times[sorted_indices[threshold]]
        training += [r for r in owned_rs if interests[0,r] < threshold_time]
        validating = [ext_rs[i] for i in sorted_indices[threshold:]]

        for t in training:
            training_matrix[uidx,t] = 1
        for v in validating:
            validating_matrix[uidx,v] = 1
        recommendation_times[uidx] = threshold_time

    comment="""
Training interests are before validating interests.
The split is as follows:
    Training: all internals before first last 1/3 externals + first 2/3 externals
    Testing: last 1/3 externals"""

    version_dir = join(data_processed_dir, version)
    tfn = join(version_dir, TRAINING_FN)
    vfn = join(version_dir, VALIDATING_FN)
    rtfn = join(version_dir, RECOMMENDATION_TIMES_FN)

    mmwrite(tfn, training_matrix, comment=comment)
    mmwrite(vfn, validating_matrix, comment=comment)
    np.save(rtfn, recommendation_times)

    return (tfn, vfn, rtfn)


if __name__ == '__main__':
    from argparse import ArgumentParser

    argparser = ArgumentParser()
    argparser.add_argument('--test', help="Is this a test or not",
                           action="store_true")
    args = argparser.parse_args()

