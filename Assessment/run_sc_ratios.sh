#!/bin/bash
# Running the percentage of recommendations in social component

for alg in TPOPSIMD SIM TPOPD
do
    echo $alg
    for K in 10 20 40
    do
        echo $K
        python in_sc_ratio.py Assessments/$alg/in_sc_$K.npy ../Data/processed/v4/validation_times_repos_per_user.pkl $K
    done
done
