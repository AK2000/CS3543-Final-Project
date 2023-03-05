import time
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from readTriples import readTriples
import sys
sys.path.insert(0,'../src/')
import spdnn
import algorithms.utils
import algorithms.GO

parser = argparse.ArgumentParser()
parser.add_argument('--neurons', default=1024, choices=[1024, 4096, 16384, 65536], help="Number of neurons for training [options: 1024, 4096, 16384, 65536], defaults to 1024.", type=int)
parser.add_argument('--layers', default=120, help="Number of layers", type=int)
parser.add_argument('--reorder', default=None, choices=[None, "GO"])
parser.add_argument('-o', '--output', help="File to write timing results to.")

args = parser.parse_args()

# Set locations of files.
basePath = '/mnt/d/UChicago/src/classes/graphs/final_project/dataset/'

inputFile = basePath + 'MNIST/sparse-images-';
categoryFile = basePath + 'DNN/neuron';
layerFile = basePath + 'DNN/neuron';

# Select DNN to run.
#Nneuron = [1024, 4096, 16384, 65536];
Nneuron = [args.neurons];
SAVECAT = False    # Overwrite truth categories.
READTSV = True    # Read input and layers from .tsv files.
READMAT = True    # Redd input and layers from .mat files.

# Select number of layers to run.
#maxLayers = 120 * [1, 4, 16];
maxLayers = [args.layers]

# Set DNN bias.
neuralNetBias = [-0.3,-0.35,-0.4,-0.45];

# Loop over each DNN.
for i in range (len(Nneuron)):
    # Load sparse MNIST data.
    if READTSV:
        # filename = inputFile + str(Nneuron[i]) + '.tsv'
        filename = f"{inputFile}{Nneuron[i]}.tsv"
        print("[INFO] Reading file: %s" %(filename))
        df = readTriples(filename)
    
    featureVectors = csr_matrix((df[2].values, (df[0].values, df[1].values)), dtype=np.int32).tocsc()
    featureVectors.resize((featureVectors.shape[0], Nneuron[i]))
    NfeatureVectors = featureVectors.shape[0]
    
# Read layers.
for j in range(len(maxLayers)):
    DNNedges = 0
    layers = []
    bias = []
    tic = time.perf_counter()
    cur_layer = 0
    next_layer = Nneuron[i]
    edge_dfs = []
    for k in range(maxLayers[j]):
        if READTSV:
            filename = f"{layerFile}{Nneuron[i]}/n{Nneuron[i]}-l{k+1}.tsv"
            # print(filename)
            df = readTriples(filename)
            df.iloc[:,0] += cur_layer
            df.iloc[:,1] += next_layer
            edge_dfs.append(df)

        DNNedges += len(df);
        cur_layer += Nneuron[i]
        next_layer += Nneuron[i]

        # TODO: Add bias

    edge_df = pd.concat(edge_dfs)
    
    readLayerTime = time.perf_counter() - tic
    readLayerRate = DNNedges/readLayerTime;

    print('[INFO] DNN neurons/layer: %d, layers: %d, edges: %d' %(Nneuron[i], maxLayers[j], DNNedges))
    print('[INFO] Read time (sec): %f, read rate (edges/sec): %f' %(readLayerTime, readLayerRate));
    print('[INFO] Max Source Neuron: %d, max desitnation neuron: %d, Calculated Max Neuron: %d' %(edge_df.iloc[:,0].max(), edge_df.iloc[:,1].max(), Nneuron[i] * (maxLayers[j]+1)))

    print('[INFO] Reordering Edges')
    tic = time.perf_counter()
    if args.reorder == "GO":
        graph = algorithms.utils.edges_to_operations_graph(edge_df, Nneuron[i])
        node_order = algorithms.GO.reorder_nodes(graph, Nneuron[i])[::-1]
    else:
        node_order = edge_df[::-1].itertuples(index=False)
    reorder_time = time.perf_counter() - tic
    print('[INFO] Time to reorder edges: %f' % reorder_time)

    tic = time.perf_counter()

    neurons = set()
    edges = []
    for row in node_order:
        activation = row[1] not in neurons
        if activation:
            neurons.add(row[1])
        edges.append(spdnn.Edge(row[0], row[1], row[2], activation))
    
    edges = edges[::-1]
    createEdgesTime = time.perf_counter() - tic
    
    print('[INFO] Created edge list. Time: %f' % createEdgesTime)

    with open("edge_order.txt", "w") as fp:
        fp.writelines(["%f %f %f\n" % (l.source, l.dest, l.weight) for l in edges])

    # Perform and time challenge
    tic = time.perf_counter()
    scores = spdnn.infer_basic(featureVectors, edges, Nneuron[i] * (maxLayers[j]+1), Nneuron[i], NfeatureVectors)  
    # scores = featureVectors
    challengeRunTime = time.perf_counter() - tic

    challengeRunRate = NfeatureVectors * DNNedges / challengeRunTime
    print('[INFO] Run time (sec): %f, run rate (edges/sec): %f' %(challengeRunTime, challengeRunRate))  

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Software Engineer: Dr. Jeremy Kepner                    
# % MIT                   
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % (c) <2019> Massachusetts Institute of Technology
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

