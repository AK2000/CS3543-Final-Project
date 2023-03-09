import time
import os
import argparse
import csv
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix

import sys
sys.path.insert(0,'./SpDNN/src')
import spdnn
from algorithms import GO
from  algorithms.EON import EON
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', help="Path to sparse network file")
    parser.add_argument('-l', '--layers', default=120, help="Number of layers to use", type=int)
    parser.add_argument('-f', '--features', help="Number of input features", type=int)
    parser.add_argument('-i', '--inputs', help="Input file or number")
    parser.add_argument('-g', '--generate', default=None, help="Algorithm to use to generate order of operations")
    parser.add_argument('--order', help="File to read/write edge ordering from", default=None)
    parser.add_argument('-o', '--output', help="File to write timing results to.")
    parser.add_argument('--infer', default=False, action="store_true", help="Peform inference")

    args = parser.parse_args()
    return args

def generate_edge_order(edge_df, algo, nfeatures, nneurons, output_path):
    print('[INFO] Reordering Edges')
    tic = time.perf_counter()
    if algo == "GO":
        graph = nx.from_pandas_edgelist(edge_df, 0, 1, 2, create_using=nx.DiGraph)
        node_order = GO.reorder_nodes(graph)[::-1]
    elif algo == "EON":
        training_log = f"training-{nfeatures}-{nneurons}.log"
        graph = nx.from_pandas_edgelist(edge_df, 0, 1, 2, create_using=nx.DiGraph)
        graph.add_nodes_from(range(edge_df[1].max()))
        node_order = EON.reorder_edges(graph, training_log_path=training_log)[::-1]
    elif algo == "None":
        node_order = edge_df[::-1].itertuples(index=False)
    else:
        print("Agorithm not recognized, Quitting!")
    reorderTime = time.perf_counter() - tic
    reorderRate = len(edge_df.index) / reorderTime
    print('[INFO] Run time: %f, Run rate: %f' % (reorderTime, reorderRate))

    neurons = set()
    edges = []
    for row in node_order:
        activation = row[1] not in neurons
        if activation:
            neurons.add(row[1])
        edges.append(spdnn.Edge(int(row[0]), int(row[1]), row[2], activation))
    edges = edges[::-1]

    with open(output_path, "w") as fp:
      fp.writelines(["%f %f %f\n" % (l.source, l.dest, l.weight) for l in edges])

    return reorderRate

def read_edge_order(filepath):
    edge_df = pd.read_csv(filepath, sep=" ", header=None)
    node_order = edge_df[::-1].itertuples(index=False)

    neurons = set()
    edges = []
    for row in node_order:
        activation = row[1] not in neurons
        if activation:
            neurons.add(row[1])
        edges.append(spdnn.Edge(int(row[0]), int(row[1]), row[2], activation))
    
    edges = edges[::-1]
    return edges

def perform_inference(inputs, edges, nneurons, nedges):
    nexamples, nfeatures = inputs.shape
    tic = time.perf_counter()
    scores = spdnn.infer_basic(inputs, edges, nneurons, nfeatures, nexamples)
    challengeRunTime = time.perf_counter() - tic
    challengeRunRate = nexamples * nedges / challengeRunTime
    print('[INFO] Run time (sec): %f, run rate (edges/sec): %f' %(challengeRunTime, challengeRunRate))

    return challengeRunRate

def main():
    args = parse_args()
    nneurons, edge_df = read_network(args.network, args.layers, args.features)

    if not args.infer:
        time = generate_edge_order(edge_df, args.generate, args.features, nneurons, args.order)
    else:
        feature_vecs = get_input_features(args.inputs, args.features)
        edges = read_edge_order(args.order)
        nedges = len(edge_df.index)
        time = perform_inference(feature_vecs, edges, nneurons, nedges)

    if args.output is not None:
        with open(args.output, 'a') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow([args.network, args.layers, args.generate, time])

if __name__ == "__main__":
    main()
