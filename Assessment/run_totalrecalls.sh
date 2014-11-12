#!/bin/bash
# Get total recalls for all passed algorithms

# $@ is the list of algorithms

vtr_fn="../Data/processed/v4/validation_times_repos_per_user.pkl"

for alg in $@
do
    echo $alg
    for K in 10 20 40
    do
        echo $K
        if [ $K -eq 10 ]
        then
            python total_recall.py Assessments/$alg/recalls_10_0.npy $vtr_fn
        fi
        if [ $K -eq 20 ]
        then
            python total_recall.py Assessments/$alg/recalls_20_1.npy $vtr_fn
        fi
        if [ $K -eq 40 ]
        then
            python total_recall.py Assessments/$alg/recalls_40_2.npy $vtr_fn
        fi
    done
done
