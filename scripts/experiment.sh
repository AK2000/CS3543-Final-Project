#!/bin/bash
for network in "radix" "random/0.1" "random/0.05" "random/0.01"; do
    for layer in "2" "4" "8" "16"; do
        for algo in "GO", "None"; do
            python SpDNN/python/runSparseDNN.py \
                        -n dataset/networks/${network} \
                        -l ${layer} \
                        -f 1024 \
                        -i dataset/inputs/sparse-images-1024.tsv \
                        --order edge_order.txt \
                        -g ${algo} \
                        -o results/reorder_times.txt

            run_cmd=python SpDNN/python/runSparseDNN.py \
                        -n dataset/networks/${network} \
                        -l ${layer} \
                        -f 1024 \
                        -i dataset/inputs/sparse-images-1024.tsv \
                        --order edge_order.txt \
                        -o results/inference_times.txt

            perf stat -e L1-dcache-loads,L1-dcache-load-misses \
                        -r 10 \
                        -x , \
                        -o results/cache_results.csv \
                        --append \
                        -- ${run_cmd} > /dev/null
                        
            exp="${network},${layer},${algo}"
            sed '${s/$/${exp}/}' results/cache_results.csv
        done
    done
done