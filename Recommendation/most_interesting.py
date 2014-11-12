#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
from os.path import abspath, join, dirname, exists
import sys
import time

import numpy as np
from scipy.io import mmread

dotdot = dirname(dirname(abspath(__file__)))
sys.path.append(join(dotdot, "Utilities"))
sys.path.append(join(dotdot, "DataProcessing"))
from recommender import Recommender
from interest_scores import get_interest_scorer
from general import timedmethod, get_matrix_before_t
from paths import VALID_REPOS_AND_TIMES
from helpers import repositories_interested_by


class MostInteresting(Recommender):
    """
        This approach is a super simple recommender based on ordering
        repositories for each user according to an interest metric
        and then recommending the top `store_atmost` of those.

        `dataset_dir`   the directory where the dataset data is stored
        `store_atmost`  max number of scores per user stored on disk
    """
    @timedmethod
    def __init__(self, interest_metric, dataset_dir, store_atmost=100):
        super(MostInteresting, self).__init__(dataset_dir)
        self.interest_metric = interest_metric
        self.store_atmost = store_atmost #we don't need to order/store all repositories
        self.scorer = get_interest_scorer(interest_metric, dataset_dir,
                                          store_atmost=store_atmost)
        self.NU, self.NR = self.scorer.NU, self.scorer.NR

        with open(join(dataset_dir, VALID_REPOS_AND_TIMES),'rb') as pf:
            valid_times_repos_list = cPickle.load(pf)
        self.prediction_times = [a[0] for a in valid_times_repos_list]

        # with open(join(dataset_dir, self.interest_metric+".txt"), "w") as f:
        #     f.write(self.tmp_dir)

        self.recommended = set() #Is used to differentiate between a run
                                 #in the current execution and a run in a
                                 #previous execution

    def __str__(self):
        """Return textual description of actual recommender"""
        return self.interest_metric

    def pad_with_rand_repos(self, top_repos, pad_size, uidx, t):
        """Returns `top_repos` + randomly chosen repos to get an array
           of size self.store_atmost where the randomly chosen repos
           are not repos uidx already interacted with at time t.
        """
        possible_padding = np.setdiff1d(np.arange(self.NR, dtype=np.int),
                                        top_repos, assume_unique=True)
        state = get_matrix_before_t(self.scorer.u_r_t, t).tocsr()
        past_repos = repositories_interested_by(uidx, state)
        possible_padding = np.setdiff1d(possible_padding, past_repos,
                                        assume_unique=True)
        np.random.shuffle(possible_padding)
        return np.append(top_repos, possible_padding[:pad_size])


    def generate_timed_recommendations(self, uidx):
        """Return 2-D array of times x store_atmost recommendations
           where times are each validation time of `uidx`.
        """
        repo_scores_t = self.scorer.get_interest_scores(uidx)
        times = self.prediction_times[uidx]
        #Assume there are interest scores only for repositories
        #the user is NOT already interested in

        user_top_repos = np.zeros((repo_scores_t.shape[0],
                                   self.store_atmost), dtype=np.int)

        for r in xrange(repo_scores_t.shape[0]):
            selected = repo_scores_t.row == r
            scores = repo_scores_t.data[selected]
            repos = repo_scores_t.col[selected]
            #Pad if necessary
            if len(scores) < self.store_atmost:
                pad_size = self.store_atmost - len(scores)
                repos = self.pad_with_rand_repos(repos, pad_size, uidx, times[r])
                scores = np.append(scores, np.zeros(pad_size, dtype=scores.dtype))

            pairs = np.array([scores, repos]).T
            np.random.shuffle(pairs)
            scores, repos = pairs.T
            args_top_rs = (-scores).argsort()[:self.store_atmost]
            user_top_repos[r] = repos[args_top_rs]

        return user_top_repos

    @timedmethod
    def recommend(self, results_fn):
        """Write to `results_fn` the number users sized list of
           2-D arrays of number validation time x self.store_atmost
           repository recommendations
        """
        if not exists(results_fn) and results_fn not in self.recommended:
            recommendations = []

            for uidx in xrange(self.NU):
                past = time.time()
                recommendations.append(self.generate_timed_recommendations(uidx))
                sys.stdout.write("\r{} / {} ({:.3f}s)".format(
                    uidx+1, self.NU, time.time()-past))
                sys.stdout.flush()

            print ""

            np.savez(results_fn, *recommendations)

            self.recommended.add(results_fn)


if __name__ == '__main__':
    from argparse import ArgumentParser

    from nose.tools import ok_, eq_

    from paths import PROCESSED_DATA_DIR
    from paths import VU_TO_I_FN, VR_TO_I_FN
    from indexer import Indexer

    argparser = ArgumentParser()
    argparser.add_argument('version', help="dataset version")
    argparser.add_argument('approach',
                help="Choose from: ICN, IAA, IRA, SCN, SAA, SRA, APOP, ASIM," \
                     "POP, CSIM-TFIDFCOS")
    args = argparser.parse_args()

    dataset_dir = join(PROCESSED_DATA_DIR, "test", args.version)
    mi = MostInteresting(args.approach, dataset_dir, store_atmost=5)
    parameters = {"K":4}
    all_recommendations = mi.recommend(parameters)

    u_to_i = Indexer(join(dataset_dir, VU_TO_I_FN))
    r_to_i = Indexer(join(dataset_dir, VR_TO_I_FN))
    # print "u_to_i",u_to_i

    eq_(len(all_recommendations), mi.NU)

    user = 1
    uidx = u_to_i[user]
    recommendations = all_recommendations[uidx]
    eq_(recommendations.shape[0], 2)
    eq_(recommendations.shape[1], 4)

    eq_(len(np.unique(recommendations[0])), 4)
    eq_(len(np.unique(recommendations[1])), 4)

    urt = mi.scorer.u_r_t.tocsr()
    for udx in xrange(len(u_to_i)):
        repo_times = urt[udx]
        for i, t in enumerate(mi.prediction_times[udx]):
            m = get_matrix_before_t(repo_times, t)
            past_repos = m.col
            try:
                ok_(not np.any(np.in1d(all_recommendations[udx][i], past_repos)))
            except:
                print "all_recommendations[{}][{}]".format(udx,i), all_recommendations[udx][i]
                print "past_repos", past_repos
                sys.exit()

    print "Tests pass"
