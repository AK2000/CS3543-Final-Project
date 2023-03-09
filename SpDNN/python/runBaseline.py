import time
import os
import argparse
import csv
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
import torch

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
    parser.add_argument('-a', '--algo', default=None, help="Library to use to perform the inference")
    parser.add_argument('-o', '--output', help="File to write timing results to.")

    args = parser.parse_args()
    return args

def scipy_inference(W, Y0):
    YMAX = 32 # set max value
    Y = Y0
    for i in range(len(W)):
        Y = Y * W[i]
        Y[Y < 0] = 0
        Y[Y > YMAX] = YMAX
    return Y

def pytorch_inference(W, Y0):
    Y = Y0
    for i in range(len(W)):
        Y = torch.matmul(Y0, W[i])

def main():
    args = parse_args()
    feature_vecs = get_input_features(args.inputs, args.features)
    nexamples = feature_vecs.shape[0]

    if args.algo == "scipy":
        nedges, layers = read_network_baseline(args.network, args.layers, args.features)
        tic = time.perf_counter()
        scores = scipy_inference(layers, feature_vecs)
        challengeRunTime = time.perf_counter() - tic

    elif args.algo == "pytorch":
        nedges, layers = read_network_baseline(args.network, args.layers, args.features)
        l = layers[0]
        layers = [torch.sparse_csr_tensor(l.indptr, l.indices, l.data, l.shape) for l in layers]
        feature_vecs = feature_vecs.astype(np.float64)
        feature_vecs = torch.sparse_csr_tensor(feature_vecs.indptr, feature_vecs.indices, feature_vecs.data)
        feature_vecs = feature_vecs.transpose(0, 1)
        tic = time.perf_counter()
        scores = pytorch_inference(layers, feature_vecs)
        challengeRunTime = time.perf_counter() - tic

    elif args.algo == "ours":
        nneurons, edge_df = read_network(args.network, args.layers, args.features)
        edge_order = edge_df[::-1].itertuples(index=False)
        neurons = set()
        edges = []
        for row in edge_order:
            activation = row[1] not in neurons
            if activation:
                neurons.add(row[1])
            edges.append(spdnn.Edge(int(row[0]), int(row[1]), row[2], activation))
        edges = edges[::-1]
        nedges = len(edge_df.index)

        tic = time.perf_counter()
        scores = spdnn.infer_basic(feature_vecs, edges, nneurons, args.features, nexamples)
        challengeRunTime = time.perf_counter() - tic


    challengeRunRate = nexamples * nedges / challengeRunTime
    print('[INFO] Run time (sec): %f, run rate (edges/sec): %f' %(challengeRunTime, challengeRunRate))
    
    if args.output is not None:
        with open(args.output, 'a') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow([args.network, args.layers, args.algo, challengeRunRate])

if __name__ == "__main__":
    main()
