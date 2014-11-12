#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build:

    the basic valid users/repositories dictionary
        <Datapath>/<version>/vu_to_i.pkl
        <Datapath>/<version>/vr_to_i.pkl
    the interest time matrix
        <Datapath>/<version>/repo_user_times.mtx
    the basic training/validating matrices
        <Datapath>/<version>/training_matrix.mtx
        <Datapath>/<version>/validating_matrix.mtx
    the recommendation times
        <Datapath>/<version>/recommendation_times.mtx
    the followerships times
        <Datapath>/<version>/followerships.mtx

valid users:
    - have at least 3 external interests
    - in followership network

valid repositories:
    - contributed to or starred by a valid user
"""

from argparse import ArgumentParser
from os.path import join, abspath, splitext
import sys
import time
from math import ceil

import numpy as np
from scipy.sparse import lil_matrix
from scipy.io import mmwrite

sys.path.append(abspath(join("..", "Utilities")))
from general import getDB
from paths import VU_TO_I_FN, VR_TO_I_FN
from paths import PROCESSED_DATA_DIR
from indexer import Indexer
from repo_user_times import generate_repo_user_times
from training_validating_rt import generate_training_validating_rt
from followerships import generate_followerships


argparser = ArgumentParser()
argparser.add_argument('version', help="dataset version")
argparser.add_argument('--test', help="Is this a test or not",
                       action="store_true")
argparser.add_argument('--split', type=float, help="training split",
                       default=2.0/3.0)
args = argparser.parse_args()

past = time.time()

db = getDB(is_test=args.test)
cursor = db.cursor()
if args.test:
    processed_data_dir = join(PROCESSED_DATA_DIR, "test")
else:
    processed_data_dir = PROCESSED_DATA_DIR

if args.version in ["v2", "v3", "v4"]:
    cursor.execute("""CREATE TEMPORARY TABLE non_artefact_interests AS
                        SELECT i.user_id, i.repository_id
                        FROM interests as i, repositories as r
                        WHERE i.repository_id = r.id AND
                              ((r.is_fork IS NULL OR r.is_fork IS FALSE) OR
                               0 < (SELECT COUNT(*)
                                    FROM starrings as s
                                    WHERE s.repository_id = r.id)
                              )
                   """)
    cursor.execute("""ANALYZE non_artefact_interests""")
    cursor.execute("""CREATE TEMPORARY TABLE at_least_3_ext as
                          SELECT u.id
                          FROM users as u
                          WHERE 3 <= (SELECT COUNT(nai.repository_id)
                                      FROM non_artefact_interests as nai,
                                           repositories as r
                                      WHERE u.id = nai.user_id AND
                                            nai.repository_id = r.id AND
                                            r.owner_id != nai.user_id)
                   """)
    cursor.execute("""ANALYZE at_least_3_ext""")
    #Get valid users - Need to make sure that each valid user follows or
    #is followed by another user with at least 3 external interests
    cursor.execute("""SELECT al31.id
                      FROM at_least_3_ext as al31,
                           at_least_3_ext as al32,
                           followerships as f
                      WHERE al31.id = f.follower_id AND
                            f.followed_id = al32.id
                      UNION
                      SELECT al31.id
                      FROM at_least_3_ext as al31,
                           at_least_3_ext as al32,
                           followerships as f
                      WHERE al31.id = f.followed_id AND
                            f.follower_id = al32.id
                   """)
    v_users = [u[0] for u in cursor.fetchall()]
    u_to_i = Indexer(join(processed_data_dir, args.version, VU_TO_I_FN))
    u_to_i.clear()
    for u in v_users:
        tmp = u_to_i[u]
    u_to_i.save()
    cursor.execute("""DROP TABLE IF EXISTS {}_users""".format(args.version))
    cursor.execute("""CREATE TABLE {}_users
                      (id INTEGER PRIMARY KEY,
                       FOREIGN KEY (id) REFERENCES users)
                   """.format(args.version))
    cursor.executemany("""INSERT INTO {}_users (id) VALUES (%s)""".format(
                        args.version), ((u,) for u in v_users))

    #Get valid repositories
    cursor.execute("""SELECT DISTINCT nai.repository_id
                      FROM non_artefact_interests as nai,
                           {0}_users as u
                      WHERE nai.user_id = u.id
                   """.format(args.version))
    v_repositories = [r[0] for r in cursor.fetchall()]
    r_to_i = Indexer(join(processed_data_dir, args.version, VR_TO_I_FN))
    r_to_i.clear()
    for r in v_repositories:
        tmp = r_to_i[r]
    r_to_i.save()
    cursor.execute("""DROP TABLE IF EXISTS {}_repositories""".format(args.version))
    cursor.execute("""CREATE TABLE {}_repositories
                      (id INTEGER PRIMARY KEY,
                       FOREIGN KEY (id) REFERENCES repositories)
                   """.format(args.version))
    cursor.executemany("""INSERT INTO {}_repositories (id) VALUES (%s)""".format(args.version),
                       ((r,) for r in v_repositories))
    db.commit()

    # r_to_i = Indexer(join(processed_data_dir, args.version, VR_TO_I_FN))
    # u_to_i = Indexer(join(processed_data_dir, args.version, VU_TO_I_FN))

    print "Number of valid users:", len(u_to_i)
    print "Number of valid repositories:", len(r_to_i)

    r_u_t_fn = generate_repo_user_times(args.version, r_to_i, u_to_i,
                                        is_test=args.test)
    tr_fn, v_fn, rt_fn = generate_training_validating_rt(
                            args.version, r_to_i, u_to_i, r_u_t_fn,
                            args.split, is_test=args.test)
    f_fn = generate_followerships(args.version, u_to_i, is_test=args.test)

present = time.time()
print "Wall clock time: %.3f s" % (present - past)

if args.test:
    from nose.tools import ok_, eq_
    from scipy.io import mmread

    print "u_to_i",u_to_i
    print "r_to_i",r_to_i
    ok_( [1,2,3,5,6] == u_to_i.keys() )
    ok_( range(1,12) == r_to_i.keys() )
    eq_(mmread(r_u_t_fn).nnz, 26)
    eq_(mmread(tr_fn).nnz, 20)
    eq_(mmread(v_fn).nnz, 7)
    rt = np.load(rt_fn)
    ok_( np.all(set(rt) == set([160000,198000,300000,202000,210000])) )
    foll = mmread(f_fn).tocsr()
    eq_(foll.nnz, 6)
    eq_(foll[u_to_i[1],u_to_i[2]], 122000)
    ok_( np.all((foll.sum(0).A1+foll.sum(1).A1) >= 1) )


    print "Tests Pass"
