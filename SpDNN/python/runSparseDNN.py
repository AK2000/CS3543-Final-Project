import time
import os
import argparse
import csv
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import sys
sys.path.insert(0,'./SpDNN/src')
import spdnn
import algorithms.utils
import algorithms.GO
from readTriples import readTriples

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

def read_network(network_directory, layers, nfeatures):
    tic = time.perf_counter()
    cur_layer = 0
    next_layer = nfeatures
    edge_dfs = []
    for k in range(layers):
        filename = f"{network_directory}/n{nfeatures}-l{k+1}.tsv"
        df = readTriples(filename)
        layer_size = df.iloc[:,1].max()+1

        df.iloc[:,0] += cur_layer
        df.iloc[:,1] += next_layer
        
        cur_layer = next_layer
        next_layer += layer_size
        edge_dfs.append(df)

    edge_df = pd.concat(edge_dfs)

    readLayerTime = time.perf_counter() - tic
    readLayerRate = len(df.index)/readLayerTime;

    print('[INFO] DNN neurons: %d, layers: %d, edges: %d' %(next_layer, layers, len(df.index)))
    print('[INFO] Read time (sec): %f, read rate (edges/sec): %f' %(readLayerTime, readLayerRate))
    return next_layer, edge_df

def get_input_features(inputs, nfeatures):
    if os.path.exists(inputs):
        # Read the inputs from a file
        print("[INFO] Reading file: %s" %(inputs))
        df = readTriples(inputs)
        featureVectors = csr_matrix((df[2].values, (df[0].values, df[1].values)), dtype=np.int32).tocsc()
        featureVectors.resize((featureVectors.shape[0], nfeatures))

    else:
        featureVectors = np.random.rand(int(inputs), nfeatures)

    return featureVectors

def generate_edge_order(edge_df, algo, nfeatures, nneurons, output_path):
    print('[INFO] Reordering Edges')
    tic = time.perf_counter()
    if algo == "GO":
        graph = algorithms.utils.edges_to_operations_graph(edge_df, nfeatures, nneurons)
        node_order = algorithms.GO.reorder_nodes(graph, nfeatures)[::-1]
    elif algo == "None":
        node_order = edge_df[::-1].itertuples(index=False)
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
