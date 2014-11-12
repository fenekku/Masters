#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
All the paths used for the project
"""

import os
from os.path import abspath, join, dirname

BASE_PATH = dirname(dirname(abspath(__file__)))

DATA_PATH = join(BASE_PATH, "Data")
DATA_CACHE_PATH = join(DATA_PATH, ".cache")

RAW_DATA_PATH = join(DATA_PATH, "raw")
READMES_DIR = join(RAW_DATA_PATH, "readmes")
TEST_READMES_DIR = join(RAW_DATA_PATH, "test", "readmes")

PROCESSED_DATA_DIR = join(DATA_PATH, "processed")
# PROCESSED_DATA_CACHE_PATH = join(PROCESSED_DATA_DIR, ".cache")

GRAPHCHI_PATH = join(BASE_PATH, "ThirdParty", "graphchi-cpp")

PREDICTION_RESULTS_PATH = join(BASE_PATH, "Prediction", "Results")
# PREDICTION_CACHE_PATH = join(BASE_PATH, "Prediction", ".cache")

def iter_users_dirs():
    """Iterator over the user ids their directory path
    """
    top_directory = join(DYNAMIC_FEATURES_PATH, "social")
    for cohort in os.listdir(top_directory):
        cohort_path = join(top_directory, cohort)
        for user in os.listdir(cohort_path):
            yield int(user), join(cohort_path, user)


#Filenames
TRAINING_FN = "training_matrix.mtx"
VALIDATING_FN = "validating_matrix.mtx"
TIMED_INTERESTS_FN = "repo_user_times.mtx"
RECOMMENDATION_TIMES_FN = "recommendation_times.npy"
PREDICTION_TIMES_FN = "prediction_times.pkl"
VALID_REPOS_AND_TIMES = "valid_repos_and_times.pkl"
FOLLOWERSHIPS_FN = "followerships.mtx"
POPULARITY_FN = "POP_scores.mtx"
VU_TO_I_FN = "vu_to_i.pkl"
#filename used by Indexers for valid users - the directory distinguishes
#btw datasets
VR_TO_I_FN = "vr_to_i.pkl"
#their associated repositories - same thing
TF_IDF_FN = "tf-idf.mtx"

# POSITIVE_TRAINING_FN = join(PROCESSED_DATA_DIR,
#                             "training_matrix_2.mtx")
# POSITIVE_VALIDATING_FN = join(PROCESSED_DATA_DIR,
#                               "validating_matrix_2.mtx")
# UU_FREQUENCIES_FN = join(PROCESSED_DATA_DIR,
#                          "useruser_frequencies.mtx")

# INTEREST_SCORES_FN = join(PROCESSED_DATA_DIR,"scores_")
# USAGE_SIMILARITY_DIR = join(PROCESSED_DATA_DIR,"usage_similarity_scores")


#Test data to use when testing
TEST_PROCESSED_DATA_PATH = join(PROCESSED_DATA_DIR, "test")
TEST_TRAINING_FN = join(TEST_PROCESSED_DATA_PATH, "training_matrix_1.mtx")
TEST_VALIDATING_FN = join(TEST_PROCESSED_DATA_PATH, "validating_matrix_1.mtx")
TEST_POSITIVE_TRAINING_FN = join(TEST_PROCESSED_DATA_PATH,
                                 "training_matrix_2.mtx")
TEST_POSITIVE_VALIDATING_FN = join(TEST_PROCESSED_DATA_PATH,
                                   "validating_matrix_2.mtx")
TEST_UU_FREQUENCIES_FN = join(TEST_PROCESSED_DATA_PATH,
                              "useruser_frequencies.mtx")
TEST_TIMED_INTERESTS_FN = join(TEST_PROCESSED_DATA_PATH,
                               "timed_interests.mtx")
TEST_RECOMMENDATION_TIMES_FN = join(TEST_PROCESSED_DATA_PATH,
                                   "recommendation_times.npy")
TEST_FOLLOWERSHIPS_FN = join(TEST_PROCESSED_DATA_PATH, "followerships.mtx")

TEST_PROCESSED_DATA_CACHE_PATH = join(TEST_PROCESSED_DATA_PATH, ".cache")

TEST_POPULARITY_FN = join(TEST_PROCESSED_DATA_PATH,"popularity_scores.mtx")
TEST_INTEREST_SCORES_FN = join(TEST_PROCESSED_DATA_PATH,"scores_")
TEST_USAGE_SIMILARITY_DIR = join(TEST_PROCESSED_DATA_PATH,"usage_similarity_scores")
