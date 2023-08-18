#!/bin/bash

features=1024
for layer in "2" "4" "8" "120"; do
    for algo in "scipy" "pytorch" "ours"; do
        run_cmd="python SpDNN/python/runBaseline.py \
                    -n dataset/networks/radix \
                    -a ${algo} \
                    -l ${layer} \
                    -f ${features} \
                    -i dataset/inputs/sparse-images-1024.tsv \
                    -o baseline_results/inference_times.csv"

        perf stat -e L1-dcache-loads,L1-dcache-load-misses \
                    -x , \
                    -o baseline_results/temp.csv \
                    --append \
                    -- ${run_cmd} > /dev/null

        exp=",${network},${layer},${algo},${features}"
        sed '/#.*$/d' baseline_results/temp.csv > baseline_results/temp_2.csv
        sed '/^\s*$/d' baseline_results/temp_2.csv > baseline_results/temp.csv
        sed "s|\$|${exp}|" baseline_results/temp.csv >> baseline_results/cache_results.csv
        rm baseline_results/temp_2.csv
        rm baseline_results/temp.csv

	echo .
    done
    echo "Finished layer ${layer}"
done
