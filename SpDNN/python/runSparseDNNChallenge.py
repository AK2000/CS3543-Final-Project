import time
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from readTriples import readTriples
import sys
sys.path.insert(0,'../src/')
import spdnn

parser = argparse.ArgumentParser()
parser.add_argument('--neurons', default=1024, choices=[1024, 4096, 16384, 65536], help="Number of neurons for training [options: 1024, 4096, 16384, 65536], defaults to 1024.", type=int)

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
maxLayers = [120];

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
    # Read in true categories.
    if not SAVECAT:
        filename = f"{categoryFile}{Nneuron[i]}-l{maxLayers[j]}-categories.tsv"
        # print(filename)
        trueCategories = np.genfromtxt(filename)
        # FIXING THE INDEXING: True Categories are +1
        trueCategories = trueCategories - 1
        
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

    tic = time.perf_counter()

    neurons = set()
    edges = []
    for row in edge_df[::-1].itertuples():
        activation = row[2] not in neurons
        if activation:
            neurons.add(row[2])
        edges.append(spdnn.Edge(row[1], row[2], row[3], activation))
    
    edges = edges[::-1]
    createEdgesTime = time.perf_counter() - tic
    
    print('[INFO] Created edge list. Time: %f' % createEdgesTime)

    # Perform and time challenge
    tic = time.perf_counter()
    scores = spdnn.infer_basic(featureVectors, edges, Nneuron[i] * (maxLayers[j]+1), Nneuron[i], NfeatureVectors, 5000)  
    # scores = featureVectors
    challengeRunTime = time.perf_counter() - tic

    challengeRunRate = NfeatureVectors * DNNedges / challengeRunTime
    print('[INFO] Run time (sec): %f, run rate (edges/sec): %f' %(challengeRunTime, challengeRunRate));

    # Compute categories from scores.
    scores_sum = scores.sum(axis=1)
    categories, col = scores_sum.nonzero()
    val = scores_sum

    if SAVECAT:
    #   StrFileWrite(sprintf('%d\n',categories),[categoryFile num2str(Nneuron(i)) '-l' num2str(maxLayers(j)) '-categories.tsv']);
        pass
    else:
        categoryDiff = csr_matrix((np.ones_like(trueCategories), (trueCategories, np.zeros_like(trueCategories))), shape=(NfeatureVectors, 1)) - csr_matrix((np.ones_like(categories), (categories, np.zeros_like(categories))), shape=(NfeatureVectors, 1))
        print("[INFO] Non-zero category difference: %d" %(categoryDiff.count_nonzero()))
        if (categoryDiff.count_nonzero()):
            print('[INFO] Challenge FAILED');
        else:
            print('[INFO] Challenge PASSED');    

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Software Engineer: Dr. Jeremy Kepner                    
# % MIT                   
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % (c) <2019> Massachusetts Institute of Technology
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

