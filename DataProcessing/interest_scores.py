#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate interest score matrix
"""

import cPickle
import os
from array import array
from os.path import join, abspath, exists, dirname
import sys
import time
from itertools import izip
import logging

import arrow
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.io import mmread, mmwrite
from sklearn.preprocessing import normalize

sys.path.append(abspath(join("..", "Utilities")))
from general import get_matrix_before_t, get_matrix_from_to_t, timedfunction
from general import COHORT_SIZE
from popularity_scores import get_popularity_scores
from interest_similarity_scores import get_interest_similarity_scores
from content_similarity_scores import get_content_similarity_scores
from paths import FOLLOWERSHIPS_FN, PREDICTION_TIMES_FN, TIMED_INTERESTS_FN
from paths import POPULARITY_FN
from helpers import neighbours, repositories_interested_by, users_interested_in


class InterestScorer(object):
    """Approach to predict interest_scores"""

    def __init__(self, interest_metric, dataset_dir, store_atmost):
        self.interest_metric = interest_metric
        self.dataset_dir = dataset_dir
        self.store_atmost = store_atmost
        self.u_r_t = mmread(join(dataset_dir, TIMED_INTERESTS_FN)).transpose()
        self.prediction_times = cPickle.load(open(join(dataset_dir, PREDICTION_TIMES_FN),"rb"))
        self.NU, self.NR = self.u_r_t.shape

    def get_interests_array(self):
        return array('d')

    def get_interest_scores(self, uidx):
        """Return COO matrix of shape times x all repositories
           of recommendation scores
        """

        scores_fn = join(self.dataset_dir, self.interest_metric, str(uidx)+"_scores.mtx")

        if exists(scores_fn):
            return mmread(scores_fn)
        elif not exists(dirname(scores_fn)):
            os.makedirs(dirname(scores_fn))

        #force copy of pointer values in loops
        timeslots = array('I')
        all_rs = array('I')
        interests = self.get_interests_array()

        for i, t in enumerate(self.prediction_times[uidx]):
            state = get_matrix_before_t(self.u_r_t, t).tocsr()
            rs, scores = self.get_related_repositories_and_scores(uidx,
                                                                  state,
                                                                  at_t=t)
            args_top_repos = (-scores).argsort()[:self.store_atmost]

            for ri in args_top_repos:
                score = scores[ri]
                if not np.isclose(score, 0.0):
                    timeslots.append(i)
                    all_rs.append(rs[ri])
                    interests.append(score)

        shape = (len(self.prediction_times[uidx]), self.NR)
        interests_scores = coo_matrix( (interests, (timeslots, all_rs)),
                                       shape=shape)
        mmwrite(scores_fn, interests_scores)

        return interests_scores


class SimilarScorer(InterestScorer):
    """Computes interest scores based on interest similarity
       Adapted from "Amazon.com Recommendations Item-to-Item
       Collaborative Filtering" by Greg Linden, Brent Smith, and Jeremy York
    """
    def __init__(self, dataset_dir, store_atmost):
        super(SimilarScorer, self).__init__("ASIM", dataset_dir, store_atmost)

    def get_related_repositories_and_scores(self, uidx, state, at_t=None):
        ns = neighbours(uidx, self.interest_metric, state)
        u_rs = repositories_interested_by(uidx, state)
        rs = np.setdiff1d(repositories_interested_by(ns, state),
                          u_rs, assume_unique=True)
        scores = get_interest_similarity_scores(u_rs, rs, state)
        return rs, scores

class PopularSimilarScorer(InterestScorer):
    """Scores similar repositories according to popularity
       Adapted from "Amazon.com Recommendations Item-to-Item
       Collaborative Filtering" by Greg Linden, Brent Smith, and Jeremy York
    """
    def __init__(self, dataset_dir, store_atmost):
        super(PopularSimilarScorer, self).__init__("APOP", dataset_dir, store_atmost)

    def get_interests_array(self):
        return array('H')

    def get_related_repositories_and_scores(self, uidx, state, at_t=None):
        ns = neighbours(uidx, self.interest_metric, state)
        rs = np.setdiff1d(repositories_interested_by(ns, state),
                          repositories_interested_by(uidx, state),
                          assume_unique=True)
        scores = get_popularity_scores(state, rs)
        return rs, scores


def AA(a):
    """computes the inverse of a"""
    if a != 0:
        return 1./np.log1p(a)
    else:
        return a

AA_func = np.vectorize(AA)

class LinkScorer(InterestScorer):
    """Link scorer based on and adapted from  Lu and Zhou's
       "Role of Weak Ties in Link Prediction of Complex Networks"
    """
    def __init__(self, interest_metric, dataset_dir, store_atmost):
        super(LinkScorer, self).__init__(interest_metric, dataset_dir,
                                         store_atmost)

    def get_scores(self, matrix, user_neighbours):
        """Return interest_score_{user, repo} according to user neighbourhood.
        """
        if self.interest_metric[1] == "C" or self.interest_metric[2] == "C":
            return matrix[user_neighbours,:].sum(axis=0).A1

        elif self.interest_metric[1] == "A" or self.interest_metric[2] == "A":
            tmp = matrix[user_neighbours,:].astype(np.float)
            if tmp.nnz == 0:
                return tmp.sum(axis=0).A1
            newvalues = AA_func(tmp.sum(axis=1).A1)
            for i,nv in enumerate(newvalues):
                #row i
                tmp.data[tmp.indptr[i]:tmp.indptr[i+1]] = nv
            return tmp.sum(axis=0).A1

        elif self.interest_metric[1] == "R" or self.interest_metric[2] == "R":
            return normalize(matrix[user_neighbours,:].astype(np.float), norm='l1').sum(axis=0).A1

        else:
            raise Exception("Invalid interest metric")

class SocialSharedInterestScorer(LinkScorer):
    """Social scorer: computes interest scores based on followership
       network neighbours
    """
    def __init__(self, interest_metric, dataset_dir, store_atmost):
        super(SocialSharedInterestScorer, self).__init__(interest_metric,
                                                         dataset_dir,
                                                         store_atmost)
        self.followerships = mmread(join(dataset_dir, FOLLOWERSHIPS_FN))

    def get_related_repositories_and_scores(self, uidx, state, at_t=None):
        social_matrix = get_matrix_before_t(self.followerships, at_t).tocsr()
        ns = neighbours(uidx, self.interest_metric, social_matrix)
        state_ = state.astype(bool).astype(np.int)
        scores = self.get_scores(state_, ns)
        scores[repositories_interested_by(uidx, state_)] = 0
        return np.arange(self.NR, dtype=np.int), scores


class TimedSocialSharedInterestScorer(LinkScorer):
    """Computes interest scores based on interest of other users with
       previous common interest on a time window
    """
    def __init__(self, interest_metric, dataset_dir, store_atmost):
        super(TimedSocialSharedInterestScorer, self).__init__(interest_metric,
                                                              dataset_dir,
                                                              store_atmost)
        self.followerships = mmread(join(dataset_dir, FOLLOWERSHIPS_FN))
        if interest_metric[-1].lower() == 'm':
            self.earlier = {'months':-1}
        elif interest_metric[-1].lower() == 'w':
            self.earlier = {'weeks':-1}
        elif interest_metric[-1].lower() == 'd':
            self.earlier = {'days':-1}
        else:
            print "TimedSocialSharedInterestScorer Error: 'window'={} is not in ['m', 'w', 'd']".format(window)
            sys.exit()

    def get_related_repositories_and_scores(self, uidx, G_I_t, at_t=None):
        G_F_t = get_matrix_before_t(self.followerships, at_t).tocsr()
        ns = neighbours(uidx, self.interest_metric, G_F_t)
        G_I_interval = get_matrix_from_to_t(G_I_t,
                                            arrow.get(at_t).replace(**self.earlier).timestamp,
                                            at_t).astype(bool).astype(np.int).tocsr()
        G_I_t_ = G_I_t.astype(bool).astype(np.int)
        scores = self.get_scores(G_I_interval, ns)
        scores[repositories_interested_by(uidx, G_I_t_)] = 0
        rs = scores.nonzero()[0]
        scores = scores[rs]
        return rs, scores


class SharedInterestScorer(LinkScorer):
    """Computes interest scores based on interest of other users with
       previous common interest
    """
    def __init__(self, interest_metric, dataset_dir, store_atmost):
        super(SharedInterestScorer, self).__init__(interest_metric,
                                                   dataset_dir,
                                                   store_atmost)

    def get_related_repositories_and_scores(self, uidx, state, at_t=None):
        state_ = state.astype(bool).astype(np.int)
        ns = neighbours(uidx, self.interest_metric, state_)
        scores = self.get_scores(state_, ns)
        scores[repositories_interested_by(uidx, state_)] = 0
        return np.arange(self.NR, dtype=np.int), scores

class TimedSharedInterestScorer(LinkScorer):
    """Computes interest scores based on interest of other users with
       previous common interest on a time window
    """
    def __init__(self, interest_metric, dataset_dir, store_atmost):
        super(TimedSharedInterestScorer, self).__init__(interest_metric,
                                                   dataset_dir,
                                                   store_atmost)
        if interest_metric[-1].lower() == 'm':
            self.earlier = {'months':-1}
        elif interest_metric[-1].lower() == 'w':
            self.earlier = {'weeks':-1}
        elif interest_metric[-1].lower() == 'd':
            self.earlier = {'days':-1}
        else:
            print "TimedSharedInterestScorer Error: 'window'={} is not in ['m', 'w', 'd']".format(window)
            sys.exit()

    def get_related_repositories_and_scores(self, uidx, G_I_t, at_t=None):
        G_I_interval = get_matrix_from_to_t(G_I_t,
                                            arrow.get(at_t).replace(**self.earlier).timestamp,
                                            at_t).astype(bool).astype(np.int).tocsr()
        G_I_t_ = G_I_t.astype(bool).astype(np.int)
        ns = neighbours(uidx, self.interest_metric[1:], G_I_t_)
        scores = self.get_scores(G_I_interval, ns)
        scores[repositories_interested_by(uidx, G_I_t_)] = 0
        return np.arange(self.NR, dtype=np.int), scores


class PopularityScorer(InterestScorer):
    """Scores according to popularity"""
    def __init__(self, dataset_dir, store_atmost):
        super(PopularityScorer, self).__init__("POP", dataset_dir, store_atmost)

    def get_interests_array(self):
        return array('H')

    def get_related_repositories_and_scores(self, uidx, state, at_t=None):
        scores = get_popularity_scores(state)
        scores[repositories_interested_by(uidx, state)] = 0
        return np.arange(self.NR, dtype=np.int), scores

class TrendingScorer(InterestScorer):
    """Scores according to temporal popularity"""
    def __init__(self, interest_metric, dataset_dir, store_atmost):
        super(TrendingScorer, self).__init__(interest_metric, dataset_dir, store_atmost)
        if interest_metric[-1].lower() == 'm':
            self.earlier = {'months':-1}
        elif interest_metric[-1].lower() == 'w':
            self.earlier = {'weeks':-1}
        elif interest_metric[-1].lower() == 'd':
            self.earlier = {'days':-1}
        else:
            print "TrendingScorer Error: 'window'={} is not in ['m', 'w', 'd']".format(window)
            sys.exit()

    def get_interests_array(self):
        return array('H')

    def get_related_repositories_and_scores(self, uidx, state, at_t=None):
        _state = get_matrix_from_to_t(state,
                                      arrow.get(at_t).replace(**self.earlier).timestamp,
                                      at_t)
        scores = get_popularity_scores(_state)
        scores[repositories_interested_by(uidx, state)] = 0
        return np.arange(self.NR, dtype=np.int), scores


class MarkovScorer(InterestScorer):
    """Scores according to Markov chains of interests"""
    def __init__(self, interest_metric, dataset_dir, store_atmost):
        super(MarkovScorer, self).__init__(interest_metric, dataset_dir, store_atmost)
        logging.basicConfig(filename='Markov.log',level=logging.DEBUG)
        logging.info("Markov launched "+arrow.utcnow().format('YYYY-MM-DD HH:mm:ss'))
        self.back_weight = float(interest_metric[3:])

    def get_interests_array(self):
        return array('f')

    def get_related_repositories_and_scores(self, uidx, G_I_t, at_t=None):
        ui_coo = G_I_t[uidx].tocoo()
        scores = np.zeros(self.NR)

        if len(ui_coo.data) == 0:
            # logging.debug('Uidx: {} at time {} has no data'.format(uidx, at_t))
            return np.zeros(1), np.zeros(1)
        else:
            prev_ridx = ui_coo.col[np.argmax(ui_coo.data)]

        related_users = users_interested_in(prev_ridx, G_I_t)

        for user in related_users:
            G_I_t_u_coo = G_I_t[user].tocoo()
            ordered_args = np.argsort(G_I_t_u_coo.data)
            ordered_interests = G_I_t_u_coo.col[ordered_args]

            arg_prev_ridx = (ordered_interests == prev_ridx).nonzero()[0][0]

            if arg_prev_ridx-1 >= 0:
                prev_prev_ridx = ordered_interests[arg_prev_ridx-1]
                scores[prev_prev_ridx] += self.back_weight #previous less important than next

            if arg_prev_ridx+1 < len(ordered_interests):
                next_prev_ridx = ordered_interests[arg_prev_ridx+1]
                scores[next_prev_ridx] += 1 #next more important

        scores[repositories_interested_by(uidx, G_I_t)] = 0
        rs = scores.nonzero()[0]
        scores = scores[rs]
        return rs, scores

def get_interest_scorer(interest_metric, dataset_dir, store_atmost=None):
    """Return `interest_metric` scorer
    """

    if interest_metric.startswith("S"):
        return SocialSharedInterestScorer(interest_metric, dataset_dir, store_atmost)

    elif interest_metric.startswith("I"):
        return SharedInterestScorer(interest_metric, dataset_dir, store_atmost)

    elif interest_metric.startswith("TI"):
        return TimedSharedInterestScorer(interest_metric, dataset_dir, store_atmost)

    elif interest_metric.startswith("TASIM"):
        from timed_sim_scorer import TimedSimilarScorer
        return TimedSimilarScorer(interest_metric, dataset_dir, store_atmost)

    elif interest_metric.startswith("TS"):
        return TimedSocialSharedInterestScorer(interest_metric, dataset_dir, store_atmost)

    elif interest_metric == "ASIM":
        return SimilarScorer(dataset_dir, store_atmost)

    elif interest_metric == "APOP":
        return PopularSimilarScorer(dataset_dir, store_atmost)

    elif interest_metric.startswith("TAPOP"):
        from timed_popsim_scorer import TimedPopularSimilarScorer
        return TimedPopularSimilarScorer(interest_metric, dataset_dir, store_atmost)

    elif interest_metric == "POP":
        return PopularityScorer(dataset_dir, store_atmost)

    elif interest_metric.startswith("TPOP"):
        return TrendingScorer(interest_metric, dataset_dir, store_atmost)

    elif interest_metric.startswith("Mkv"):
        return MarkovScorer(interest_metric, dataset_dir, store_atmost)

    elif interest_metric.startswith("TMkv"):
        from timed_markov_scorer import TimedMarkovScorer
        return TimedMarkovScorer(interest_metric, dataset_dir, store_atmost)

    elif interest_metric.startswith("MMkv"):
        from multistep_markov_scorer import MultiMarkovScorer
        return MultiMarkovScorer(interest_metric, dataset_dir, store_atmost)


if __name__ == "__main__":
    from argparse import ArgumentParser

    from nose.tools import ok_

    from paths import VU_TO_I_FN, VR_TO_I_FN, PROCESSED_DATA_DIR
    from indexer import Indexer

    argparser = ArgumentParser()
    argparser.add_argument('version', help="dataset version")
    argparser.add_argument('approach',
                help="Choose from: ICN, IAA, IRA, SCN, SAA, SRA, APOP, ASIM," \
                     "POP, TPOP{d,w,m}")
    argparser.add_argument("--test", action='store_true', help='test?')
    args = argparser.parse_args()

    if args.test:
        dataset_dir = join(PROCESSED_DATA_DIR, "test", args.version)
    else:
        dataset_dir = join(PROCESSED_DATA_DIR, args.version)

    u_to_i = Indexer(join(dataset_dir, VU_TO_I_FN))
    r_to_i = Indexer(join(dataset_dir, VR_TO_I_FN))

    if args.test:
        user = 1
        uidx = u_to_i[user]

        scorer = get_interest_scorer(args.approach, dataset_dir, store_atmost=4)

        i = scorer.get_interest_scores(uidx)
        import shutil
        shutil.rmtree(join(dataset_dir, args.approach))

        print "r_to_i"
        print r_to_i
        print "u_to_i"
        print u_to_i

        ia = i.toarray()
        print "interest scores"
        print ia
        escores = np.zeros((2,len(r_to_i)))

        if args.approach == "SCN":
            escores[0,r_to_i[5]] = 1
            escores[1,r_to_i[4]] = 1
            escores[1,r_to_i[9]] = 1
            ok_( np.all(ia == escores) )

        elif args.approach == "ICN":
            escores[0,r_to_i[5]] = 1
            escores[1,r_to_i[4]] = 2
            # escores[1,r_to_i[7]] = 1 eliminated by store_atmost
            escores[1,r_to_i[8]] = 1
            escores[1,r_to_i[9]] = 1
            escores[1,r_to_i[10]] = 1
            ok_( np.allclose(ia, escores) )

        elif args.approach == "IRA":
            escores[0,r_to_i[5]] = 0.5
            escores[1,r_to_i[4]] = 9./20.
            # escores[1,r_to_i[7]] = 1./5. eliminated by store_atmost
            escores[1,r_to_i[8]] = 1./5.
            escores[1,r_to_i[9]] = 1./4.
            escores[1,r_to_i[10]] = 1./5.
            ok_( np.allclose(ia, escores) )

        elif args.approach == "IAA":
            escores[0,r_to_i[5]] = 1./np.log1p(2.)
            escores[1,r_to_i[4]] = 1./np.log1p(5.) + 1./np.log1p(4.)
            # escores[1,r_to_i[7]] = 1./5. eliminated by store_atmost
            escores[1,r_to_i[8]] = 1./np.log1p(5.)
            escores[1,r_to_i[9]] = 1./np.log1p(4.)
            escores[1,r_to_i[10]] = 1./np.log1p(5.)
            ok_( np.allclose(ia, escores) )

        elif args.approach == "ASIM":
            escores[0,r_to_i[5]] = 1./2.
            escores[1,r_to_i[4]] = 2./np.sqrt(6)
            escores[1,r_to_i[7]] = 1./np.sqrt(3)
            escores[1,r_to_i[8]] = 1./np.sqrt(6)
            escores[1,r_to_i[9]] = 1./2.
            # escores[1,r_to_i[10]] = 1./np.sqrt(6) eliminated by store_atmost
            ok_( np.allclose(ia, escores) )

        elif args.approach == "POP":
            escores[0,r_to_i[5]] = 2
            escores[0,r_to_i[8]] = 2
            escores[0,r_to_i[6]] = 1
            escores[0,r_to_i[3]] = 1
            # escores[0,r_to_i[4]] = 1
            # escores[0,r_to_i[7]] = 1

            escores[1,r_to_i[8]] = 2
            escores[1,r_to_i[10]] = 2
            escores[1,r_to_i[4]] = 2
            escores[1,r_to_i[9]] = 2
            # escores[1,r_to_i[3]] = 2
            # escores[1,r_to_i[6]] = 1 eliminated by store_atmost
            # escores[1,r_to_i[7]] = 1 eliminated by store_atmost
            ok_( np.allclose(ia, escores) )

        elif args.approach == "APOP":
            escores[0,r_to_i[5]] = 2
            escores[1,r_to_i[4]] = 2
            # escores[1,r_to_i[7]] = 1 eliminated by store_atmost
            escores[1,r_to_i[8]] = 2
            escores[1,r_to_i[9]] = 2
            escores[1,r_to_i[10]] = 2

            ok_( np.allclose(ia, escores) )

        elif args.approach == "TPOPD":
            user = 6
            uidx = u_to_i[user]
            i = scorer.get_interest_scores(uidx)
            import shutil
            shutil.rmtree(join(dataset_dir, args.approach))
            ia = i.toarray()
            print "ia", ia

            escores[0,r_to_i[11]] = 3
            escores[0,r_to_i[9]] = 2
            escores[0,r_to_i[6]] = 1
            escores[0,r_to_i[1]] = 1
            # escores[0,r_to_i[2]] = 1 eliminated by store_atmost
            # escores[0,r_to_i[3]] = 1 eliminated by store_atmost
            #Rest is all seros

            ok_( np.allclose(ia, escores) )

        elif args.approach == "TIRAD":
            escores[0,r_to_i[5]] = 0.5
            escores[1,r_to_i[4]] = 9./20.
            # escores[1,r_to_i[7]] = 1./5. eliminated by store_atmost
            escores[1,r_to_i[8]] = 1./5.
            escores[1,r_to_i[9]] = 1./4.
            escores[1,r_to_i[10]] = 1./5.
            ok_( np.allclose(ia, escores) )

        elif args.approach == "Mkv0.5":
            user = 6
            uidx = u_to_i[user]
            print "uidx", uidx
            i = scorer.get_interest_scores(uidx)
            import shutil
            if exists(join(dataset_dir, args.approach)):
                shutil.rmtree(join(dataset_dir, args.approach))
            ia = i.toarray()
            print "ia", ia

            escores[0,r_to_i[11]] = 0.5

            escores[1,r_to_i[1]] = 0.5
            escores[1,r_to_i[11]] = 1
            #Rest is all zeros

            ok_( np.allclose(ia, escores) )


        print "Tests pass"

    else:
        print "Generating "+ args.approach + " scores..."
        NU = len(u_to_i)
        past1 = time.time()
        scorer = get_interest_scorer(args.approach, dataset_dir, store_atmost=100)
        for uidx in xrange(NU):
            past = time.time()
            interests_scores = scorer.get_interest_scores(uidx)
            sys.stdout.write("\r{} / {} ({:.3f}s)".format(
                    uidx+1, NU, time.time()-past))
            sys.stdout.flush()

        print ""
        print "{} ({:.3f}s)".format(uidx+1, time.time()-past1)
