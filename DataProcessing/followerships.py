#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build interested users x interested users sparse matrix of timed
followerships.
"""

from os.path import join, abspath
import sys

import arrow
import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import mmwrite

sys.path.append(abspath(join("..", "Utilities")))
from paths import PROCESSED_DATA_DIR, FOLLOWERSHIPS_FN
from general import getDB
from general import timedfunction


@timedfunction
def generate_followerships(version, u_to_i, is_test=False):
    """Return filename of file containing transcribed followerships"""
    if is_test:
        data_processed_dir = join(PROCESSED_DATA_DIR, "test")
    else:
        data_processed_dir = PROCESSED_DATA_DIR

    frs = []
    fds = []
    ts = []

    cursor = getDB(is_test=is_test).cursor("f")
    cursor.execute("""SELECT f.follower_id, f.followed_id, f.occurred_on
                      FROM followerships as f,
                           {0} as u1,
                           {0} as u2
                      WHERE u1.id = f.follower_id AND
                            u2.id = f.followed_id
                   """.format(version+"_users"))

    for fr, fd, t in cursor:
        frs.append(u_to_i[fr])
        fds.append(u_to_i[fd])
        ts.append(arrow.get(t).to('utc').timestamp)

    cursor.close()

    followerships = coo_matrix((ts, (frs, fds)), dtype=np.int_, shape=(len(u_to_i),len(u_to_i)))
    fn = join(data_processed_dir, version, FOLLOWERSHIPS_FN)
    mmwrite(fn, followerships, comment="followership_{u,u} utc timestamps")

    return fn


if __name__ == '__main__':
    from argparse import ArgumentParser
    argparser = ArgumentParser()
    argparser.add_argument('--test', help="Is this a test or not",
                           action="store_true")
    args = argparser.parse_args()

    # generate_followerships()