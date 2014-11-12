#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Indexing table classes. These tables are used to map a user or repository
id to a unique index among all users/repositories considered.

This is necessary for matrix/array indexing everywhere.
"""

import os
from os.path import exists, dirname
import cPickle

from bidict import bidict   #nice little bidirectional dict


class Indexer(object):
    """Table that indexes various ids to sequential integers and
       vice-versa.
    """
    def __init__(self, indexfilepath):
        self.indexfilepath = indexfilepath
        self._id_to_i = bidict()
        self._id_index = 0

        if exists(self.indexfilepath):
            with open(self.indexfilepath, "rb") as f:
                self._id_to_i = cPickle.load(f)
                self._id_index = len(self._id_to_i)

    def __getitem__(self, key):
        """Return the index of `key` or a new index if `key` isn't indexed.
        """
        if key not in self._id_to_i:
            self._id_to_i[key] = self._id_index
            self._id_index += 1
        return self._id_to_i[key]

    def __len__(self):
        """Number of elements in the indexer"""
        return len(self._id_to_i)

    def __iter__(self):
        """Iterate over the keys of the indexer"""
        return iter(self._id_to_i)

    def __str__(self):
        """String representation of mapping"""
        return "\n"+"\n".join(["{} -> {}".format(id_, i) for id_,i in self._id_to_i.items()])

    def keys(self):
        """Return all keys"""
        return self._id_to_i.keys()

    def r(self, key):
        """Return the index corresponding to `key`
        """
        return self._id_to_i[:key]

    def save(self, path=None):
        """Save indexing table to disk
        """
        if path is None:
            path = self.indexfilepath
        if not exists(dirname(path)):
            os.makedirs(dirname(path))
        with open(path, "wb") as f:
            cPickle.dump(self._id_to_i, f, cPickle.HIGHEST_PROTOCOL)

    def clear(self):
        """Empties the indexing table
        """
        self._id_to_i.clear()
        self._id_index = 0


if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys
    from os.path import join
    from nose.tools import eq_
    from general import getDB2
    from paths import PROCESSED_DATA_CACHE_PATH

    argparser = ArgumentParser()
    argparser.add_argument('--test', help="Is this a test or not",
                           action="store_true")
    args = argparser.parse_args()

    if args.test:
        from paths import TEST_VU1_TO_I_FN, TEST_VR1_TO_I_FN
        vu1_to_i = Indexer(TEST_VU1_TO_I_FN)
        vu1_to_i.clear()
        vu1_to_i[331272]
        vu1_to_i[557971]
        vu1_to_i[411212]
        vu1_to_i[678987]
        vu1_to_i[348245]
        vu1_to_i[672134]
        vu1_to_i.save()

        vr1_to_i = Indexer(TEST_VR1_TO_I_FN)
        vr1_to_i.clear()
        vr1_to_i[400102]
        vr1_to_i[312433]
        vr1_to_i[500111]
        vr1_to_i[1111235]
        vr1_to_i[27899]
        vr1_to_i[102304]
        vr1_to_i[422168]
        vr1_to_i[817452]
        vr1_to_i[139007]
        vr1_to_i[921125]
        vr1_to_i.save()

        print "Generated valid users and valid repositories indexing tables"

        sys.exit()

    #Fill u_to_i and test
    u_to_i = Indexer(join(PROCESSED_DATA_CACHE_PATH, "u_to_i.pkl"))
    db = getDB2()
    cursor = db.cursor("user_id_iterator")
    cursor.execute("""SELECT id FROM users ORDER BY id ASC""")
    for uid, in cursor: #returned values are in tuple so need to unpack
        a = u_to_i[uid]
    cursor.close()
    u_to_i.save()
    eq_(uid-1, u_to_i[uid])
    eq_(uid, u_to_i.r(u_to_i[uid]))

    u_to_i = Indexer(join(PROCESSED_DATA_CACHE_PATH, "u_to_i.pkl"))
    eq_(uid-1, u_to_i[uid])

    #Fill ir_to_i
    ir_to_i = Indexer(join(PROCESSED_DATA_CACHE_PATH, "ir_to_i.pkl"))
    cursor = db.cursor("interest_repo_id_iterator")
    cursor.execute("""SELECT rid FROM
                      (SELECT DISTINCT fc.repository_id as rid
                       FROM first_contributions as fc,
                            active_users as au
                       WHERE fc.contributor_id = au.user_id
                       UNION
                       SELECT DISTINCT w.repository_id as rid
                       FROM watchings as w,
                            active_users as au,
                            contributed_repositories as cr
                       WHERE w.repository_id = cr.id AND
                             w.watcher_id = au.user_id) as tmp
                       ORDER BY rid ASC""")
    for rid, in cursor: #returned values are in tuple so need to unpack
        a = ir_to_i[rid]
    ir_to_i.save()
    eq_(rid, ir_to_i.r(ir_to_i[rid]))

    print "All tests pass"

