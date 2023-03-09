#!/bin/bash

layer=16
features=1024
for sparsity in "0.05" "0.1" "0.05" "0.01"; do
    for algo in "scipy" "pytorch" "ours"; do
        run_cmd="python SpDNN/python/runBaseline.py \
                    -n dataset/networks/random/${sparsity} \
                    -a ${algo} \
                    -l ${layer} \
                    -f ${features} \
                    -i dataset/inputs/sparse-images-1024.tsv \
                    -o baseline_results/inference_times.txt"

        perf stat -e L1-dcache-loads,L1-dcache-load-misses \
                    -r 10 \
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
    done
done