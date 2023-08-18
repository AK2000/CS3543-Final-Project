#!/bin/bash
for network in "radix" "random/0.01" "random/0.05" "random/0.1"; do
    for layer in "2" "4" "8" "16"; do
        for features in "1024"; do
            for algo in "GO" "None"; do
                if python SpDNN/python/runSparseDNN.py \
                            -n dataset/networks/${network} \
                            -l ${layer} \
                            -f ${features} \
                            -i ${features} \
                            --order edge_order.txt \
                            -g ${algo} \
                            -o results/reorder_times.txt; then

                    run_cmd="python SpDNN/python/runSparseDNN.py \
                                -n dataset/networks/${network} \
                                --infer \
                                -g ${algo} \
                                -l ${layer} \
                                -f ${features} \
                                -i ${features} \
                                --order edge_order.txt \
                                -o results/inference_times.txt"

                    perf stat -e L1-dcache-loads,L1-dcache-load-misses \
                                -r 10 \
                                -x , \
                                -o results/temp.csv \
                                --append \
                                -- ${run_cmd} > /dev/null

                    exp=",${network},${layer},${algo},${features}"
                    sed '/#.*$/d' results/temp.csv > results/temp_2.csv
                    sed '/^\s*$/d' results/temp_2.csv > results/temp.csv
                    sed "s|\$|${exp}|" results/temp.csv >> results/cache_results.csv
                    rm results/temp_2.csv
                    rm results/temp.csv
                else
                    echo "Command failed"
                fi
            done
        done
    done
done
