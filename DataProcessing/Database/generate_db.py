#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extract data from the archives and store them in the database

   This script assumes that the observed data is in the 2011-2012 (<February)
   time period. Changes in the API format beyond this date is incompatible
   with some of the processing done here.
"""

from argparse import ArgumentParser
import cPickle
import codecs
import gzip
from itertools import izip
import logging
import simplejson as json
from os.path import abspath, join, exists
import os
import re
import sys
import time
import traceback
#will I need to import ALL of python's standard library?!

import arrow

sys.path.append(abspath(join("..", "..", "Utilities")))
from general import getDB
from paths import DATA_PATH
from json_stream import json_string_generator

argparser = ArgumentParser(description="Converts .json.gz activity file of hourly " \
                                       "activity into a postgresql database.")
argparser.add_argument('--test', help="Is this a test or not",
                       action="store_true")
argparser.add_argument('--load_from_disk', help="Add the activities to the previous ones",
                       action="store_true")
argparser.add_argument('dir', default=None,
                        help="Directory where the .json.gz files are")

args = argparser.parse_args()

db = getDB(is_test=args.test)
cursor = db.cursor()

if args.test:
    #Insert users
    uids = range(1,9)
    gravatar_ids = ["1A","1B","1C","2A","2B","2C","3CC","4D"]
    names = ["user1","user2","user3","user4","user5","user6", "org1", "user7"]
    ts = map(lambda t: arrow.get(t).to('utc'), [122000,122000,150001,
                                                175000,121000,121000,
                                                100000,103222])
    is_org = lambda uid: True if uid == 7 else False
    gen_users = ((uid,gid,n,t.datetime,is_org(uid)) for uid, gid, n, t in
                    izip(uids,gravatar_ids, names, ts))
    cursor.execute("""DROP TABLE IF EXISTS users CASCADE""")
    cursor.execute("""CREATE TABLE users
                      (id INTEGER PRIMARY KEY,
                       gravatar_id TEXT NOT NULL,
                       name TEXT NOT NULL,
                       first_seen TIMESTAMPTZ NOT NULL,
                       is_organization BOOLEAN)
                   """)
    cursor.executemany("""INSERT INTO users
                          (id, gravatar_id, name, first_seen,
                           is_organization)
                          VALUES (%s, %s, %s, %s, %s)""", gen_users)

    #Insert repositories
    rids = range(1,13)
    oids = [1,2,3,6,6,5,6,3,5,7,2,1]
    repo_names = ["css-colorguard","awesome-go","material","go-dropbox",
                  "awesome-python","h20", "cayley", "Bootstrap",
                  "Bootstra.386", "atom", "lime", "lime"]
    cts = map(lambda t: arrow.get(t).to('utc'), [130000,140000,150001,
                                                 131000,131010,145000,
                                                 132000,155000,197000,
                                                 177000,149100,159100])
    gen_repos = ((r,o,n,t.datetime) for r,o,n,t in izip(rids,oids,repo_names,cts))
    cursor.execute("""DROP TABLE IF EXISTS repositories CASCADE""")
    cursor.execute("""CREATE TABLE repositories
                      (id INTEGER PRIMARY KEY,
                       owner_id INTEGER NOT NULL,
                       repo TEXT NOT NULL,
                       creation_time TIMESTAMPTZ NOT NULL,
                       is_fork BOOLEAN,
                       fork_of INTEGER,
                       FOREIGN KEY (owner_id) REFERENCES users,
                       FOREIGN KEY (fork_of) REFERENCES repositories)
                   """)
    cursor.executemany("""INSERT INTO repositories
                          (id, owner_id, repo, creation_time)
                          VALUES (%s, %s, %s, %s)""", gen_repos)
    cursor.execute("""UPDATE repositories
                        SET is_fork = TRUE, fork_of = 5
                        WHERE id = 2
                   """)
    cursor.execute("""UPDATE repositories
                        SET is_fork = TRUE, fork_of = 11
                        WHERE id = 12
                   """)

    #Insert followerships
    followerships = [(1,2,122000),(2,4,275000),(3,1,310000),(3,2,381121),
                     (5,6,121000),(6,5,500000),(4,8,200132), (1,3,410000)]
    gen_foll = ((fr, fd, arrow.get(t).to('utc').datetime) for fr, fd, t in followerships)
    cursor.execute("""DROP TABLE IF EXISTS followerships""")
    cursor.execute("""CREATE TABLE followerships
                      (follower_id INTEGER NOT NULL,
                       followed_id INTEGER NOT NULL,
                       occurred_on TIMESTAMPTZ NOT NULL,
                       FOREIGN KEY (follower_id) REFERENCES users,
                       FOREIGN KEY (followed_id) REFERENCES users)
                   """)
    cursor.executemany("""INSERT INTO followerships
                          (follower_id, followed_id, occurred_on)
                          VALUES (%s, %s, %s)""", gen_foll)

    #Insert first_contributions
    first_contributions = [(1, 1, 130000), (2, 1, 140000), (3, 3, 150001),
                           (4, 6, 131000), (5, 6, 131010), (11, 1, 165000),
                           (6, 5, 145000), (7, 6, 132000), (8, 3, 155000),
                           (9, 5, 197999), (9, 2, 198000), (10, 6, 177101),
                           (11, 2, 149100), (11, 5, 200101), (7, 3, 299000),
                           (3, 4, 185102), (2, 8, 145678), (12, 1, 159500)]
    gen_fc = ((r, c, arrow.get(t).to('utc').datetime) for r, c, t in first_contributions)
    cursor.execute("""DROP TABLE IF EXISTS first_contributions""")
    cursor.execute("""CREATE TABLE first_contributions
                      (contributor_id INTEGER NOT NULL,
                       repository_id INTEGER NOT NULL,
                       occurred_on TIMESTAMPTZ NOT NULL,
                       FOREIGN KEY (contributor_id) REFERENCES users,
                       FOREIGN KEY (repository_id) REFERENCES repositories)
                   """)
    cursor.executemany("""INSERT INTO first_contributions
                          (repository_id, contributor_id, occurred_on)
                          VALUES (%s, %s, %s)""", gen_fc)

    #Insert starrings
    starrings = [(2, 3, 300000), (2, 6, 210000), (3, 4, 185101),
                 (5, 1, 160000), (5, 4, 175000), (5, 5, 202000),
                 (5, 6, 131010), (8, 3, 200000), (8, 6, 157001),
                 (10, 1, 200000), (10, 3, 185000), (10, 5, 201000),
                 (11, 1, 150000), (1, 6, 310000), (4, 2, 160100),
                 (3, 8, 131050), (4, 8, 145670)]
    gen_starrings = ((r, s, arrow.get(t).to('utc').datetime) for r, s, t in starrings)
    cursor.execute("""DROP TABLE IF EXISTS starrings""")
    cursor.execute("""CREATE TABLE starrings
                      (starrer_id INTEGER NOT NULL,
                       repository_id INTEGER NOT NULL,
                       occurred_on TIMESTAMPTZ NOT NULL,
                       FOREIGN KEY (starrer_id) REFERENCES users,
                       FOREIGN KEY (repository_id) REFERENCES repositories)
                   """)
    cursor.executemany("""INSERT INTO starrings
                          (repository_id, starrer_id, occurred_on)
                          VALUES (%s, %s, %s)""", gen_starrings)

    cursor.execute("""DROP TABLE IF EXISTS interests""")
    cursor.execute('''CREATE TABLE interests AS
                        SELECT a.user_id, a.repository_id,
                               MIN(a.occurred_on) as occurred_on
                        FROM (SELECT starrer_id as user_id, repository_id, occurred_on
                              FROM starrings
                              UNION
                              SELECT contributor_id as user_id, repository_id, occurred_on
                              FROM first_contributions
                              UNION
                              SELECT owner_id as user_id, fork_of as repository_id, creation_time as occurred_on
                              FROM repositories
                              WHERE is_fork = True ) as a
                        GROUP BY a.user_id, a.repository_id
                   ''')

    db.commit()
    sys.exit()

print "Filling the database for real."

beginnning = time.time()

RETAINED_EVENTS = ["PushEvent", "ForkEvent", "FollowEvent", "WatchEvent",
                   "CreateEvent", "PublicEvent"]

retained_events_count_fn = join(DATA_PATH, "retained_events_count.pkl")
all_events_count_fn = join(DATA_PATH, "all_events_count.pkl")
malformed_events_count_fn = join(DATA_PATH, "malformed_events_count.pkl")

user_ids = set()
created_repos_ids = set()
retained_events_count = 0
all_events_count = 0
malformed_events_count = 0
next_sha_id = 1
sha_dict = {}
dumping_ground = open("dumping_ground.json", "w")

if args.load_from_disk:
    cursor.execute("""SELECT id FROM users""")
    user_ids = set([uidt[0] for uidt in cursor.fetchall()])
    cursor.execute("""SELECT id FROM repositories""")
    created_repos_ids = set([ridt[0] for ridt in cursor.fetchall()])
    retained_events_count = cPickle.load(open(retained_events_count_fn, "rb"))
    all_events_count = cPickle.load(open(all_events_count_fn, "rb"))
    malformed_events_count = cPickle.load(open(malformed_events_count_fn, "rb"))
    cursor.execute("""SELECT MAX(id) FROM shas""")
    next_sha_id = cursor.fetchone()[0]
    cursor.execute("""SELECT sha, id, rid, event_time FROM shas""")
    sha_dict = {t[0]: t[1:] for r in cursor.fetchall()}
else:
    cursor.execute("""DROP TABLE IF EXISTS repositories CASCADE""")
    cursor.execute("""DROP TABLE IF EXISTS users CASCADE""")
    cursor.execute("""DROP TABLE IF EXISTS shas CASCADE""")
    cursor.execute("""DROP TABLE IF EXISTS contributions CASCADE""")
    cursor.execute("""DROP TABLE IF EXISTS fs CASCADE""")
    cursor.execute("""DROP TABLE IF EXISTS ss CASCADE""")

# =============
# Setup Logging
# =============
logging.basicConfig(filename=__file__+'.log', level=logging.DEBUG)
logger = logging.getLogger(__file__)

p = re.compile(r"https://secure\.gravatar\.com/avatar/([0-9a-f]+)\?")

def _get_gravatar_id(actor_dict):
    """Does its darnest to get the gravatar_id
    """
    if actor_dict.get("gravatar_id", None):
        return actor_dict["gravatar_id"]
    else:
        m = p.match(actor_dict["avatar_url"])
        if m:
            return m.group(1)
        else:
            raise KeyError("gravatar_id")


def _get_user_id(actor_dict):
    """
    Returns the integer id associated with the actor represented by `actor dict`.
    """
    aid = actor_dict["id"]
    gravatar_id = _get_gravatar_id(actor_dict)

    if aid not in user_ids:
        cursor.execute("""INSERT INTO users
                          (id, gravatar_id, name, first_seen,
                           is_organization)
                          VALUES (%s, %s, %s, %s, %s)""",
                       (aid, gravatar_id,
                        actor_dict["login"], actor_dict["event_time"],
                        actor_dict["is_org"])
                      )
        user_ids.add(aid)

    return aid


def _insert_contribution(sha, actor_id, rid, event_time):
    """Add `sha` to shas table if not already seen and add a contribution
       to the contribution table of that sha to the repository `rid`
    """
    global next_sha_id
    global sha_dict

    if sha not in sha_dict:
        sha_dict[sha] = (next_sha_id, rid, event_time)
        cursor.execute("""INSERT INTO shas
                          (id, sha, contributor_id, repository_id,
                           first_seen)
                          VALUES (%s,%s,%s,%s,%s)""",
                       (next_sha_id, sha, actor_id, rid,
                        event_time)
                      )
        next_sha_id += 1

    cursor.execute("""INSERT INTO contributions
                      (sha_id, repository_id, occurred_on)
                      VALUES (%s,%s,%s)""",
                   (sha_dict[sha][0], rid, event_time)
                  )


def json_to_db(json_object):
    """ Insert json object in DB appropriately.
    """
    global all_events_count
    global retained_events_count
    global created_repos_ids

    event_type = json_object.get("type", None)
    all_events_count += 1

    # We don't even bother with some events
    if event_type not in RETAINED_EVENTS:
        return

    retained_events_count += 1

    event_time = arrow.get(json_object["created_at"]).to('utc').datetime

    # If an org is mentioned in the json object get its id
    # (and insert it in the db in passing)
    if json_object.get("org", None):
        #complete it to have all the info needed for possible creation
        json_object["org"]["event_time"] = event_time
        json_object["org"]["is_org"] = True
        org_id = _get_user_id(json_object["org"])

    actor = json_object.get("actor_attributes", None) or \
            json_object.get("actor", None)

    if not actor:
        return

    # Get actor_id
    # complete it in case it needs to be created
    actor["event_time"] = event_time
    actor["is_org"] = actor.get("type", None) == "Organization"
    actor_id = _get_user_id(actor)

    if event_type in ["CreateEvent", "ForkEvent", "PublicEvent",
                      "PushEvent", "WatchEvent"]:
        r = json_object.get("repo", None) or \
            json_object.get("repository", None)

        if r is None:
            return

        rid = r["id"]

        if event_type in ["CreateEvent", "ForkEvent", "PublicEvent"]:
            is_fork = False
            fork_of = None

            if (event_type == "CreateEvent" and
                json_object["payload"].get("ref_type", None) != "repository"):
               return

            if event_type in ["CreateEvent", "PublicEvent"]:
                if rid in created_repos_ids:
                    # This may happen when a repository is made public
                    # then private and public again: so you have 2
                    # public events for the same repository id
                    # It also may happen for a regular create event
                    # it is strange
                    return

            elif event_type == "ForkEvent":
                if rid not in created_repos_ids:
                    return

                is_fork = True
                fork_of = rid

                #rid of the newly created fork - the rid we want
                if type(json_object["payload"]["forkee"]) == int:
                    rid = json_object["payload"]["forkee"]
                else:
                    rid = json_object["payload"]["forkee"]["id"]

            if 'repo' in json_object:
                repo_name = r["name"].split("/")[1]
            else:
                repo_name = r["name"]

            cursor.execute("""INSERT INTO repositories
                              (id, owner_id, repo, creation_time,
                               is_fork, fork_of)
                              VALUES (%s,%s,%s,%s,%s,%s)
                           """,
                           (rid, actor_id, repo_name, event_time, is_fork,
                            fork_of) )

            created_repos_ids.add(rid)

        elif event_type == "PushEvent":
            if rid not in created_repos_ids:
                return

            # There are different variants for this category
            # We use shas because a closed pull request is not synonymous
            # with an accepted request in the "repo" format.
            if json_object["payload"].get("shas", None):
                # VARIANT 1
                shas = (commit[0] for commit in
                        json_object["payload"]["shas"])
            elif json_object["payload"].get("commits", None):
                # VARIANT 2
                shas = ( commit["sha"] for commit in
                         json_object["payload"]["commits"] )
            else:
                shas = []

            for sha in shas:
                # might add a sha to the db
                _insert_contribution(sha, actor_id, rid, event_time)

        elif event_type == "WatchEvent":
            if rid not in created_repos_ids:
                return

            cursor.execute("""INSERT INTO ss
                              (starrer_id, repository_id, occurred_on)
                              VALUES (%s,%s,%s)""",
                           (actor_id, rid, event_time))
            #we normalize after the fact because it is faster
            #and less cumbersome to do so

    elif event_type == "FollowEvent":
        followed = json_object["payload"]["target"]
        followed["event_time"] = event_time
        followed["is_org"] = followed.get("type", None) == "Organization"
        followed_id = _get_user_id(followed)

        cursor.execute("""INSERT INTO fs
                          (follower_id, followed_id, occurred_on)
                          VALUES (%s,%s,%s)
                       """, (actor_id, followed_id, event_time))
        #we normalize after the fact because it is faster
        #and less cumbersome to do so


def gz_to_db(gzfilename):
    """
    Unzip gzfilename and populate database with the required data
    """
    global malformed_events_count

    json_file = gzip.open(gzfilename, 'rb')

    reader = codecs.getreader("ISO-8859-2")
    json_decoded_file = reader(json_file)

    for json_string in json_string_generator(json_decoded_file.read()):
        json_object = json.loads(json_string)

        try:
            json_to_db(json_object)
        except KeyError as ke:
            malformed_events_count += 1
            continue


# Create Tables
cursor.execute("""CREATE TABLE IF NOT EXISTS users
                  (id INTEGER PRIMARY KEY,
                   gravatar_id TEXT NOT NULL,
                   name TEXT NOT NULL,
                   first_seen TIMESTAMPTZ NOT NULL,
                   is_organization BOOLEAN)
               """)

cursor.execute("""CREATE TABLE IF NOT EXISTS repositories
                  (id INTEGER PRIMARY KEY,
                   owner_id INTEGER NOT NULL,
                   repo TEXT NOT NULL,
                   creation_time TIMESTAMPTZ NOT NULL,
                   last_contribution_time TIMESTAMPTZ,
                   is_fork BOOLEAN,
                   fork_of INTEGER,
                   FOREIGN KEY (owner_id) REFERENCES users,
                   FOREIGN KEY (fork_of) REFERENCES repositories)
               """)

cursor.execute('''CREATE TABLE IF NOT EXISTS shas
                  (id INTEGER PRIMARY KEY,
                   sha TEXT NOT NULL,
                   contributor_id INTEGER NOT NULL,
                   repository_id INTEGER NOT NULL,
                   first_seen TIMESTAMPTZ NOT NULL,
                   FOREIGN KEY (contributor_id) REFERENCES users,
                   FOREIGN KEY (repository_id) REFERENCES repositories)
               ''')

cursor.execute('''CREATE TABLE IF NOT EXISTS contributions
                  (sha_id INTEGER NOT NULL,
                   repository_id INTEGER NOT NULL,
                   occurred_on TIMESTAMPTZ NOT NULL,
                   FOREIGN KEY (repository_id) REFERENCES repositories,
                   FOREIGN KEY (sha_id) REFERENCES shas)
               ''')

cursor.execute('''CREATE TABLE IF NOT EXISTS fs
                  (follower_id INTEGER NOT NULL,
                   followed_id INTEGER NOT NULL,
                   occurred_on TIMESTAMPTZ NOT NULL,
                   FOREIGN KEY (follower_id) REFERENCES users,
                   FOREIGN KEY (followed_id) REFERENCES users)
               ''')

cursor.execute('''CREATE TABLE IF NOT EXISTS ss
                  (starrer_id INTEGER NOT NULL,
                   repository_id INTEGER NOT NULL,
                   occurred_on TIMESTAMPTZ NOT NULL,
                   FOREIGN KEY (starrer_id) REFERENCES users,
                   FOREIGN KEY (repository_id) REFERENCES repositories)
               ''')
db.commit()

# Iterate over the gzipped activity files in chronological order
d = abspath(args.dir)
t = len(os.listdir(d))

for count, gzfilename in enumerate(sorted(os.listdir(d))):
    past = time.time()
    gzfilename = join(d, gzfilename)
    gz_to_db(gzfilename)
    sys.stdout.write("\r"+ str(count+1) + " / " + str(t) + " ({:.3g}s)".format(time.time()-past))
    sys.stdout.flush()
    if count % 23 == 0:
        db.commit()
db.commit()
#     logger.critical("Error! for {0}".format(gzfilename))
#     print "See", __file__+".log", "to see where you are at"
#     logger.error(""+traceback.format_exc())
#     sys.exit(-1)

print ""
print "retained_events_count / all_events_count = {} / {}".format(
    retained_events_count, all_events_count)
print "malformed_events_count / retained_events_count = {} / {}".format(
    malformed_events_count, retained_events_count)

cPickle.dump(retained_events_count, open(retained_events_count_fn, "wb"),
             cPickle.HIGHEST_PROTOCOL)
cPickle.dump(all_events_count, open(all_events_count_fn, "wb"),
             cPickle.HIGHEST_PROTOCOL)
cPickle.dump(malformed_events_count, open(malformed_events_count_fn, "wb"),
             cPickle.HIGHEST_PROTOCOL)
print "Finished converting activities"

print "Normalizing..."
past = time.time()
cursor.execute("""DROP TABLE IF EXISTS followerships CASCADE""")
cursor.execute('''CREATE TABLE followerships AS
                    SELECT follower_id, followed_id, MIN(occurred_on) as occurred_on
                    FROM fs
                    GROUP BY follower_id, followed_id
               ''')
cursor.execute("""DROP TABLE IF EXISTS starrings CASCADE""")
cursor.execute('''CREATE TABLE starrings AS
                    SELECT starrer_id, repository_id, MIN(occurred_on) as occurred_on
                    FROM ss
                    GROUP BY starrer_id, repository_id
               ''')
cursor.execute("""DROP TABLE fs""")
cursor.execute("""DROP TABLE ss""")
db.commit()
print "Done in {:.3g}s".format(time.time() - past)

print "Creating first_contributions table..."
past = time.time()
cursor.execute("""DROP TABLE IF EXISTS first_contributions""")
cursor.execute('''CREATE TABLE first_contributions AS
                    SELECT s.contributor_id, c.repository_id,
                           MIN(c.occurred_on) as occurred_on
                    FROM shas as s,
                         contributions as c
                    WHERE c.sha_id = s.id
                    GROUP BY s.contributor_id, c.repository_id
               ''')
db.commit()
print "Done in {:.3g}s".format(time.time() - past)

print "Creating interests table..."
past = time.time()
cursor.execute("""DROP TABLE IF EXISTS interests""")
cursor.execute('''CREATE TABLE interests AS
                    SELECT a.user_id, a.repository_id,
                           MIN(a.occurred_on) as occurred_on
                    FROM (SELECT starrer_id as user_id, repository_id, occurred_on
                          FROM starrings
                          UNION
                          SELECT contributor_id as user_id, repository_id, occurred_on
                          FROM first_contributions
                          UNION
                          SELECT owner_id as user_id, fork_of as repository_id, creation_time as occurred_on
                          FROM repositories
                          WHERE is_fork IS TRUE ) as a
                    GROUP BY a.user_id, a.repository_id
               ''')
db.commit()
print "Done in {:.3g}s".format(time.time() - past)


print "Indexing..."
past = time.time()
cursor.execute("""ALTER TABLE followerships ADD CONSTRAINT unique_followers UNIQUE (follower_id, followed_id)""")
cursor.execute("""ALTER TABLE starrings ADD CONSTRAINT unique_starrers UNIQUE (starrer_id, repository_id)""")
cursor.execute("""ALTER TABLE first_contributions ADD CONSTRAINT unique_firsts UNIQUE (contributor_id, repository_id)""")
cursor.execute("""ALTER TABLE interests ADD CONSTRAINT unique_interests UNIQUE (user_id, repository_id)""")
cursor.execute("""CREATE INDEX interesting_repos_index ON interests (repository_id)""")
cursor.execute("""CREATE INDEX repos_owners_index ON repositories (owner_id)""")
db.commit()
print "Done in {:.3g}s".format(time.time() - past)

print "DONE EVERYTHING in {:.3g}s".format(time.time() - beginning)
