"""
    All performance metrics are defined here.
"""

from os.path import abspath, dirname, join
import logging
import time
import sys

from scipy.sparse.csgraph import dijkstra
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

dotdot = dirname(dirname(abspath(__file__)))
sys.path.append(join(dotdot, "DataProcessing"))
sys.path.append(join(dotdot, "Utilities"))
sys.path.append(join(dotdot, "Recommendation"))
from general import timedmethod, get_matrix_before_t, imin
from helpers import repositories_interested_by
from recommender import Recommender
from generate_tops import Runner
from timed_matrix import TimedMatrix


class Metric(object):
    """Assesses the performance of a 2-D array of recommendations"""
    def __init__(self):
        pass

    def __str__(self):
        """Return textual description of actual metric"""
        return self.__class__.__name__


class AvgUserRecall(Metric):
    """Assesses avg recall over recommendations per user"""
    @timedmethod
    def __init__(self, valid_repos_list, assessments_dir):
        self.valid_repos_list = valid_repos_list
        self.NU = len(self.valid_repos_list)
        self.total_positives = np.array([len(a) for a in valid_repos_list])
        self.predicted_positives = np.zeros(self.NU)
        self.out_dir = assessments_dir
        self.fileno = 0

    @timedmethod
    def assess(self, list_timed_topks):
        """ Performs the assessment of all users at all times

            `list_timed_topks` NU-sized list of arrays of shape times x K repositories
            where rows are ordered chronologically
        """

        for uidx in xrange(self.NU):
            valid_repos = self.valid_repos_list[uidx]
            timed_topks = list_timed_topks[uidx]
            predictions = np.in1d(valid_repos, timed_topks)
            self.predicted_positives[uidx] = predictions.sum()

        recalls = self.predicted_positives / self.total_positives
        K = list_timed_topks[0].shape[1]
        np.save(join(self.out_dir, "recalls_{}_{}".format(K, self.fileno)), recalls)
        self.fileno += 1

        return np.mean(recalls)


class AvgUserUnserendipity(Metric):
    """Assesses avg unserendipity over all users"""
    def __init__(self, list_validation_times, interest_graph, assessments_dir):
        logging.basicConfig(filename="AvgUserUnserendipity_performance_"+str(time.time())+".log",level=logging.DEBUG)
        self.list_validation_times = list_validation_times
        self.NU = len(self.list_validation_times)
        self.G_I = interest_graph #repos x users
        self.out_dir = assessments_dir
        self.fileno = 0

    @timedmethod
    def assess(self, list_timed_topks):
        """ Performs the assessment of all users at all times
            `list_timed_topks` NU-sized list of arrays of shape times x K repositories
            where rows are ordered chronologically
        """
        past = time.time()
        user_unserendipities = np.array([self.unserendipity(uidx, list_timed_topks[uidx])
                                            for uidx in xrange(self.NU)])
        logging.info("AvgUserUnserendipity: user_unserendipities: "+str(time.time()-past))
        K = list_timed_topks[0].shape[1]

        past = time.time()
        np.save(join(self.out_dir, "unsrdp_{}_{}".format(K, self.fileno)), user_unserendipities)
        logging.info("AvgUserUnserendipity: np.save(user_unserendipities): "+str(time.time()-past))
        self.fileno += 1
        return np.mean( user_unserendipities )

    def unserendipity(self, uidx, timed_topks):
        """average cos similarity between previously seen repositories and
           recommended repositories averaged over time
        """
        validation_times = self.list_validation_times[uidx]
        avg_cos_sim_times = np.ones(len(validation_times))

        for i, t in enumerate(validation_times):
            G_I_t = get_matrix_before_t(self.G_I, t).tocsr()
            G_I_t = G_I_t.astype(bool).astype(np.int)
            R_u_t = timed_topks[i]
            N_u_G_I_t = repositories_interested_by(uidx, G_I_t.transpose())
            if len(N_u_G_I_t) == 0:
                #eight users have same training and validating times
                #we ignore those users
                continue
            cos_sim = cosine_similarity(G_I_t[N_u_G_I_t,:], G_I_t[R_u_t,:])
            avg_cos_sim_times[i] = np.mean(cos_sim)

        return np.mean(avg_cos_sim_times)


class AvgUserSocialDistance(Metric):
    """Avg social distance per recommended repository
    """
    @timedmethod
    def __init__(self, validation_times_per_user, G_I_fn, G_F_fn, assessments_dir):
        self.validation_times_per_user = validation_times_per_user
        self.NU = len(self.validation_times_per_user)
        self.G_I = TimedMatrix(G_I_fn)
        self.G_F = TimedMatrix(G_F_fn)
        self.out_dir = assessments_dir
        self.number_insc_per_user = np.zeros(self.NU, dtype=int)

    @timedmethod
    def assess(self, list_timed_topks):
        """Returns avg distance of recommended elements

           Parameters
           ----------
           list_timed_topks     (number users)-sized list of 2D arrays
                                of shape (number validation times, K)

           Returns
           -------
           Avg of average distances
        """
        avg_scsdist_per_user = np.array([self.avg_scdistance(uidx, list_timed_topks[uidx])
                                         for uidx in xrange(self.NU)])
        K = list_timed_topks[0].shape[1]
        np.save(join(self.out_dir, "avg_sdist_per_user_{}".format(K)), avg_scsdist_per_user)
        np.save(join(self.out_dir, "in_sc_per_user_{}".format(K)), self.number_insc_per_user)
        self.number_insc_per_user[:] = 0
        return np.mean(avg_scsdist_per_user)

    def avg_scdistance(self, uidx, recommended_repos_per_time):
        """Average over time over recommended repos of the social distance between
           uidx and each of the closest member of uidx's social component
           that has shown interest in the considered repo

           Parameters
           ----------
           uidx                             considered user idx
           recommended_repos_per_time       ndarray (number validation times, K)
        """
        times = self.validation_times_per_user[uidx]
        avg_social_distances = np.array([ np.mean(self._get_social_distances(uidx, t, recommended_repos_per_time[i]))
                                          for i,t in enumerate(times) ])
        return np.mean(avg_social_distances)

    def _get_social_distances(self, uidx, t, recommended_repos):
        G_F_t = self.G_F.at_t(t)
        G_I_t = self.G_I.at_t(t)
        uidx_to_others_distances = dijkstra(G_F_t, indices=uidx,
                                            unweighted=True,
                                            directed=False)
        uidx_to_others_distances[uidx] = np.inf #for later cleaning
        social_distances = np.array([ imin(uidx_to_others_distances[G_I_t[r].tocoo().col])
                                          for r in recommended_repos ])
        insc_recommendations = np.logical_not(np.isposinf(social_distances))
        self.number_insc_per_user[uidx] += insc_recommendations.sum()
        insc_users = np.logical_not(np.isposinf(uidx_to_others_distances))

        for i in xrange(len(social_distances)):
            if social_distances[i] == np.inf:
                if np.min(uidx_to_others_distances) == np.inf:
                    social_distances[i] = self.NU-1
                else:
                    social_distances[i] = np.max(uidx_to_others_distances[insc_users])

        return social_distances


class AvgUserSrdp(Metric):
    """Avg user Srdp as defined in 'Serendipitous Personalized Ranking for Top-N Recommendation'
       by Lu et al 2012
    """
    @timedmethod
    def __init__(self, valid_repos_list, pm_dir, assessments_dir):
        logging.basicConfig(filename="AvgUserSrdp_performance_"+str(time.time())+".log",level=logging.DEBUG)
        self.valid_repos_list = valid_repos_list
        self.NU = len(self.valid_repos_list)
        self.total_positives = np.array([len(a) for a in valid_repos_list])
        self.predicted_positives = np.zeros(self.NU)
        self.out_dir = assessments_dir
        self.fileno = 0
        self.pm_dir = pm_dir
        r_fn = Runner.run_results_fn(pm_dir, 0)
        self.pm_repos_list = Recommender.load_recommendations(r_fn)

    @timedmethod
    def assess(self, list_timed_topks):
        """ Performs the assessment of all users at all times

            `list_timed_topks` NU-sized list of arrays of shape times x K repositories
            where rows are ordered chronologically
        """
        K = list_timed_topks[0].shape[1]

        past = time.time()

        for uidx in xrange(self.NU):
            valid_repos = self.valid_repos_list[uidx]
            pred_repos = list_timed_topks[uidx]
            pm_repos = self.pm_repos_list[uidx][:,:K]
            unexpected_hits = 0

            for idx in xrange(valid_repos.size):
                vr = valid_repos[idx]
                prs = pred_repos[idx]
                pmrs = pm_repos[idx]
                urs = np.setdiff1d(prs, pmrs)
                unexpected_hits += np.in1d(vr, urs).sum()

            self.predicted_positives[uidx] = unexpected_hits

        logging.info("AvgUserSrdp: predicted_positives loop: "+str(time.time()-past))

        srdps = self.predicted_positives / self.total_positives
        past = time.time()
        np.save(join(self.out_dir, "srdps_{}_{}".format(K, self.fileno)), srdps)
        logging.info("AvgUserSrdp: np.save(srdps): "+str(time.time()-past))
        self.fileno += 1

        return np.mean(srdps)


if __name__ == '__main__':
    from argparse import ArgumentParser
    import cPickle
    import os
    from os.path import join, exists
    import time

    from ioutils import bmwrite


    argparser = ArgumentParser()
    argparser.add_argument('metric', help="Which metric to test among" \
                                          "recall, unserendipity, sdist, srdp")
    argparser.add_argument('recommendations_fn', help="filename of recommendations")
    argparser.add_argument('G_I_ru_tm_fn', help="Interest graph (repositories, users) filename")
    argparser.add_argument('vrt_fn', help="Validation repositories and times filename")
    argparser.add_argument('Ks', help="Top Ks", type=int, nargs='+')
    #Needed if doing sdist
    argparser.add_argument('--G_F_fn', help="Followership graph (users, users) filename")
    argparser.add_argument('--dir', help="Where to store the generated assessments",
                           default=join(os.getcwd(),"Assessments") )
    argparser.add_argument('--test', help="is a test?", action="store_true")
    args = argparser.parse_args()

    recommendations_fn = abspath(args.recommendations_fn)
    recommendations = Recommender.load_recommendations(recommendations_fn)
    G_I_ru_tm_fn = abspath(args.G_I_ru_tm_fn)
    G_F_fn = abspath(args.G_F_fn)
    dataset_dir = dirname(G_I_ru_tm_fn)
    out_dir = abspath(args.dir)

    if not exists(out_dir):
        os.makedirs(out_dir)

    vrt_fn = abspath(args.vrt_fn)
    with open(vrt_fn, "rb") as pf:
        valid_times_repos_list = cPickle.load(pf)
    valid_times_list = [a[0] for a in valid_times_repos_list]
    valid_repos_list = [a[1] for a in valid_times_repos_list]

    if args.test:
        from paths import VU_TO_I_FN

        from nose.tools import eq_, ok_

        from indexer import Indexer


        u_to_i = Indexer(join(dataset_dir, VU_TO_I_FN))
        uidx = u_to_i[6]

        if args.metric == "recall":
            #NEED TO REDO
            # recall = AvgUserRecall(valid_repos_list)
            # result = recall.assess(stubbed_predicted_repos)
            # eq_(result, np.array([1./2.,1.,0.,1.,0.]).mean())

        elif args.metric == "unserendipity":
            #REDO
            # interest_graph = mmread(join(dataset_dir, TIMED_INTERESTS_FN))

            # unserendipity = AvgUserUnserendipity(valid_times_list, interest_graph)
            # timed_topks = stubbed_predicted_repos[uidx]
            # result = unserendipity.unserendipity(uidx, timed_topks)
            # print "result", result
            # ok_(np.isclose(result, 0.240803169))

        elif args.metric == "sdist":
            sdist = AvgUserSocialDistance(valid_times_list, G_I_ru_tm_fn,
                                          G_F_fn, out_dir)
            recs = [rec[:,:3] for rec in recommendations]
            past = time.time()
            result = sdist.avg_scdistance(uidx, recs[uidx])
            print "Wall clock time: %.3f s" % (time.time() - past)
            print "result", result
            eq_(result, 1.0)

        elif args.metric == "srdp":
            #REDO
            # pm_dir = join(BASE_PATH, "Recommendation", "Recommendations", "test", "POP")
            # srdp = AvgUserSrdp(valid_repos_list, pm_dir, assessments_dir)
            # past = time.time()

            #test POP
            # [[6, 9, 0, 2], [6, 9, 4, 2]] -> [2,3], [0,2]
            # [[1, 8, 0, 4], [5, 9, 7, 1]] -> [9, 10, 3], [3]
            # [[5, 1, 3, 0]] -> [6, 4]
            # [[8, 6, 9, 7]] -> [3, 1]
            # [[8, 7, 1, 2]] -> [0, 4]

            # result = srdp.assess(stubbed_predicted_repos)
            # print "Wall clock time: %.3f s" % (time.time() - past)
            # print "result", result
            # eq_(result, .4)

        else:
            print "{} is an incorrect metric!".format()

        print "Tests pass"

    else:
        if args.metric == "sdist":
            metric = AvgUserSocialDistance(valid_times_list, G_I_ru_tm_fn,
                                           G_F_fn, out_dir)

        avg_assessments = np.array([ metric.assess([u_recs[:,:K] for u_recs in recommendations])
                                     for K in args.Ks ])
        assessments_fn = join(out_dir, args.metric+".assessment")
        bmwrite(assessments_fn, args.metric, {}, ("K", args.Ks), avg_assessments, np.zeros(len(args.Ks)))
        print "Generated {} assessments in {}".format(args.metric, out_dir)
