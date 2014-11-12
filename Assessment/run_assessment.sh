#!/bin/bash
# Running an assessment

# $1 is the metric
# $2 is the algorithm to assess

data_dir="../Data/processed/v4"
G_I_fn="$data_dir/timed_G_I_ru.pkl"
G_F_fn="$data_dir/timed_G_F.pkl"
vtr_fn="$data_dir/validation_times_repos_per_user.pkl"
Run="0"
Recommendations_fn_root="../Recommendation/Recommendations/$2/"

if [ "$2" = "ASIM" ]
then
    Run="3"
fi

if [ "$2" = "SCN" ]
then
    Run="8"
fi

python metrics.py "$1" "${Recommendations_fn_root}run${Run}.npz" $G_I_fn $vtr_fn 10 20 40 --G_F_fn="$G_F_fn" --dir="Assessments/$2/"

# echo POP
# python new_assessKs.py v4 $1 ../Recommendation/Recommendations/POP/run0.npz 10 20 40 --dir=Assessment/Assessments/
# echo TPOPM
# python new_assessKs.py v4 $1 ../Recommendation/Recommendations/TPOPM/run0.npz 10 20 40 --dir=Assessment/Assessments/
# echo SIM
# python new_assessKs.py v4 $1 ../Recommendation/Recommendations/ASIM/run3.npz 10 20 40 --dir=Assessment/Assessments/
# echo POPSIM
# python new_assessKs.py v4 $1 ../Recommendation/Recommendations/APOP/run0.npz 10 20 40 --dir=Assessment/Assessments/
# echo ICN
# python new_assessKs.py v4 $1 ../Recommendation/Recommendations/ICN/run0.npz 10 20 40 --dir=Assessment/Assessments/
# echo IAA
# python new_assessKs.py v4 $1 ../Recommendation/Recommendations/IAA/run0.npz 10 20 40 --dir=Assessment/Assessments/
# echo IRA
# python new_assessKs.py v4 $1 ../Recommendation/Recommendations/IRA/run3.npz 10 20 40 --dir=Assessment/Assessments/
# echo SCN
# python assessKs.py v4 $1 ../Recommendation/Recommendations/SCN/run8.npz 10 20 40 --dir=Assessment/Assessments/
# echo SAA
# python new_assessKs.py v4 $1 ../Recommendation/Recommendations/SAA/run0.npz 10 20 40 --dir=Assessment/Assessments/
# echo SRA
# python new_assessKs.py v4 $1 ../Recommendation/Recommendations/SRA/run0.npz 10 20 40 --dir=Assessment/Assessments/
# echo MKV
# python new_assessKs.py v4 $1 ../Recommendation/Recommendations/Mkv/run1.npz 10 20 40 --dir=Assessment/Assessments/

