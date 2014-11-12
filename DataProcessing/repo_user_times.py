#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build sparse matrix of dimension: valid repos x valid users
of EarliestInterest_{r,u}
                    number user
    number repos | EarliestInterest_{u,r}

This shape is by design. It makes Entropy and Coverage faster.

"""

from os.path import join, abspath
import sys

import arrow
import numpy as np
from scipy.sparse import lil_matrix, coo_matrix
from scipy.io import mmwrite

sys.path.append(abspath(join("..", "Utilities")))
from general import getDB
from paths import PROCESSED_DATA_DIR, TIMED_INTERESTS_FN
from general import timedfunction


@timedfunction
def generate_repo_user_times(version, r_to_i, u_to_i, is_test=False):
    """Function called to generate repo_user_times.mtx
    """
    if is_test:
        processed_data_dir = join(PROCESSED_DATA_DIR, "test")
    else:
        processed_data_dir = PROCESSED_DATA_DIR

    us = []
    rs = []
    ts = []

    cursor = getDB(is_test=is_test).cursor("u_r_t")
    cursor.execute("""SELECT i.user_id, i.repository_id, i.occurred_on
                      FROM interests as i,
                           {0} as vu,
                           {1} as vr
                      WHERE i.user_id = vu.id AND i.repository_id = vr.id
                   """.format(version+"_users", version+"_repositories"))

    for u, r, t in cursor:
        us.append(u_to_i[u])
        rs.append(r_to_i[r])
        ts.append(arrow.get(t).to('utc').timestamp)

    cursor.close()

    r_u_time = coo_matrix( (ts, (rs, us)), dtype=np.int_ )

    comment="""Earliest interest_{r,u} utc timestamps"""

    fn = join(processed_data_dir, version, TIMED_INTERESTS_FN)
    mmwrite(fn, r_u_time, comment=comment)

    return fn


# @timedfunction
# def test_lil(version, r_to_i, u_to_i, is_test=False):
#     users_table = version+"_users"
#     repositories_table = version+"_repositories"
#     r_u_time = lil_matrix((len(r_to_i), len(u_to_i)), dtype=np.int_)
#     print "len(r_to_i)",len(r_to_i)
#     print "len(u_to_i)",len(u_to_i)
#     cursor = getDB(is_test=is_test).cursor("u_r_t")

#     cursor.execute("""SELECT i.user_id, i.repository_id, i.occurred_on
#                       FROM interests as i,
#                            {0} as vu,
#                            {1} as vr
#                       WHERE i.user_id = vu.id AND i.repository_id = vr.id
#                       LIMIT 20000
#                    """.format(users_table, repositories_table))

#     for uid, rid, t in cursor:
#         r_u_time[r_to_i[rid], u_to_i[uid]] = arrow.get(t).to('utc').timestamp

#     cursor.close()


# @timedfunction
# def test_coo1(version, r_to_i, u_to_i, is_test=False):
#     users_table = version+"_users"
#     repositories_table = version+"_repositories"

#     db = getDB(is_test=is_test)
#     cursor = db.cursor("u")
#     cursor.execute("""SELECT i.user_id
#                       FROM interests as i,
#                            {0} as vu,
#                            {1} as vr
#                       WHERE i.user_id = vu.id AND i.repository_id = vr.id
#                       LIMIT 10000
#                    """.format(users_table, repositories_table))
#     us = np.array([u_to_i[t[0]] for t in cursor], dtype=np.int_)
#     cursor.close()

#     cursor = db.cursor("r")
#     cursor.execute("""SELECT i.repository_id
#                       FROM interests as i,
#                            {0} as vu,
#                            {1} as vr
#                       WHERE i.user_id = vu.id AND i.repository_id = vr.id
#                       LIMIT 10000
#                    """.format(users_table, repositories_table))
#     rs = np.array([r_to_i[t[0]] for t in cursor], dtype=np.int_)
#     cursor.close()

#     cursor = db.cursor("t")
#     cursor.execute("""SELECT i.occurred_on
#                       FROM interests as i,
#                            {0} as vu,
#                            {1} as vr
#                       WHERE i.user_id = vu.id AND i.repository_id = vr.id
#                       LIMIT 10000
#                    """.format(users_table, repositories_table))
#     ts = np.array([arrow.get(t[0]).to('utc').timestamp for t in cursor],
#                   dtype=np.int_)
#     cursor.close()

#     r_u_time = coo_matrix( (ts, (rs, us)), dtype=np.int_ )


# @timedfunction
# def test_coo2(version, r_to_i, u_to_i, is_test=False):
#     users_table = version+"_users"
#     repositories_table = version+"_repositories"
#     us = []
#     rs = []
#     ts = []
#     db = getDB(is_test=is_test)
#     cursor = db.cursor("urt")
#     cursor.execute("""SELECT i.user_id, i.repository_id, i.occurred_on
#                       FROM interests as i,
#                            {0} as vu,
#                            {1} as vr
#                       WHERE i.user_id = vu.id AND i.repository_id = vr.id
#                       LIMIT 20000
#                    """.format(users_table, repositories_table))
#     for u, r, t in cursor:
#         us.append(u_to_i[u])
#         rs.append(r_to_i[r])
#         ts.append(arrow.get(t).to('utc').timestamp)

#     cursor.close()

#     r_u_time = coo_matrix( (ts, (rs, us)), dtype=np.int_ )



if __name__ == '__main__':
    from paths import VU_TO_I_FN
    from paths import VR_TO_I_FN
    from indexer import Indexer

    from argparse import ArgumentParser

    argparser = ArgumentParser()
    argparser.add_argument('--test', help="Is this a test or not",
                           action="store_true")
    argparser.add_argument('version', help="dataset version")
    args = argparser.parse_args()

    if args.test:
        processed_data_dir = join(PROCESSED_DATA_DIR, "test")
    else:
        processed_data_dir = PROCESSED_DATA_DIR

    u_to_i = Indexer(join(processed_data_dir, args.version, VU_TO_I_FN))
    r_to_i = Indexer(join(processed_data_dir, args.version, VR_TO_I_FN))

    test_coo1(args.version, r_to_i, u_to_i, is_test=args.test)
    test_coo2(args.version, r_to_i, u_to_i, is_test=args.test)
    test_lil(args.version, r_to_i, u_to_i, is_test=args.test)

