#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate content similarity score matrix
"""

from os.path import join, abspath, exists
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmread, mmwrite
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(abspath(join("..", "Utilities")))
from paths import TF_IDF_FN
from general import timedfunction


@timedfunction
def get_content_similarity_scores(readmes, dataset_dir, profile="tfidf",
                                  similarity="cos"):
    """Return CSR matrix of similarity_{r,r} for all r in `readmes`.

       `dataset_dir`      the directory where the similarity scores are
       `profile`    bool or tfidf
       `similarity` cos or ijd (inverse Jacquard Distance)
    """
    if profile == "tfidf":
        sim_fn = join(dataset_dir, TF_IDF_FN)

    if exists(sim_fn):
        return mmread(sim_fn).tocsr()

    if profile == "bool":
        #readme_words = COUNTVECTORIZER readmes
        pass
    else:
        tfidf = TfidfVectorizer(input='file', #sublinear_tf=True,
                                max_df=0.5, stop_words='english',
                                decode_error="ignore")
        #max_df=0.5: if a word occurs in more than half of the readmes it is
        #            ignored
        readme_words = tfidf.fit_transform(readmes)

    if similarity == "cos":
        similarity_scores = csr_matrix(cosine_similarity(readme_words))
    else:
        # similarity_scores = csr_matrix(ijd(readme_words))
        pass

    mmwrite(sim_fn, similarity_scores, comment=profile+"_"+similarity+"_similarity_{r,r}")
    return similarity_scores


if __name__ == "__main__":
    import time

    from nose.tools import ok_

    sys.path.append(abspath(join("..", "Utilities")))
    from general import get_matrix_before_t
    from paths import TEST_TIMED_INTERESTS_FN, TEST_RECOMMENDATION_TIMES_FN
    from paths import TEST_SIMILARITY_DIR as similarity_dir

    ti = mmread(TEST_TIMED_INTERESTS_FN)
    lt = np.load(TEST_RECOMMENDATION_TIMES_FN)
    user = 4

    print lt[user] #200101

    past = time.time()

    interests = get_matrix_before_t(ti.transpose(), lt[user]).tocsr()
    s = get_usage_similarity_scores(user, interests,
                              join(similarity_dir, "cos_" + str(user)))
    print repr(s)
    s10 = s[10].toarray()[0]
    es10 = np.array([1./np.sqrt(3), 2./np.sqrt(6), 0, 0, 1./3.,
                     0, 0, 0, 0, 1./3., 0])
    print "s10:", s10
    print "es10:", es10

    ok_( np.allclose(s10, es10) )

    print "Wall clock time: {:.3f} s".format(time.time() - past)
    print "Tests pass"