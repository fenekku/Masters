#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This script generates the top K recommendations
"""

import os
from os.path import exists, join

class Runner(object):
    """Runs a recommendation algorithm multiple times
    """
    def __init__(self, recommender, results_dir):
        self.recommender = recommender
        self.results_dir = join(results_dir, str(recommender))
        if not exists(self.results_dir):
            os.makedirs(self.results_dir)

    @staticmethod
    def run_results_fn(directory, run):
        """Returns the filename of the file containing the results of `run`
        """
        return join(directory, "run"+str(run)+".npz")

    def run(self, runs):
        """Run runs times
        """
        for run in xrange(runs):
            results_fn = Runner.run_results_fn(self.results_dir, run)
            print "Generating user recommendations for run", run
            self.recommender.recommend(results_fn)

        print "Generated {} recommendations in {}".format(
                self.recommender, self.results_dir)


# Command line interface to generate recommendations here
if __name__ == '__main__':
    from argparse import ArgumentParser
    import os
    from os.path import abspath, join, exists, dirname
    import sys
    import time

    dotdot = dirname(dirname(abspath(__file__)))
    sys.path.append(join(dotdot, "Utilities"))

    from paths import PROCESSED_DATA_DIR
    from most_interesting import MostInteresting


    argparser = ArgumentParser()
    argparser.add_argument('version', help="dataset version")
    argparser.add_argument('algorithm',
                help="Choose from: ICN, IAA, IRA, SCN, SAA, SRA, POPSIM, SIM," \
                     "POP, TPOP{D,W,M}, Mkv")
    argparser.add_argument('atmost', help="Top atmost", type=int)
    argparser.add_argument('--runs', help="Number of times to run" \
                                          "recommendation algorithm",
                            type=int, default=10)
    argparser.add_argument('--dir', help="Where to store the generated recommendations",
                           default=join(os.getcwd(),"Recommendations") )
    argparser.add_argument('--test', help="Is this a test or not",
                           action="store_true")
    args = argparser.parse_args()

    if args.test:
        print "Testing"
        dataset_dir = join(PROCESSED_DATA_DIR, "test", args.version)
    else:
        print "Not testing"
        dataset_dir = join(PROCESSED_DATA_DIR, args.version)

    rdir = abspath(args.dir)

    past = time.time()

    recommender = MostInteresting(args.algorithm, dataset_dir, args.atmost)
    runner = Runner(recommender, rdir)
    runner.run(args.runs)

    present = time.time()
    print "Wall clock time: %.3f s" % (present - past)
