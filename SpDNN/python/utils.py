import time

from scipy.sparse import csr_matrix
import pandas as pd

def readTriples(fname):
    #StrFileRead: Reads a file into a string array.
    #String utility function.
    #  Usage:
    #    s = StrFileRead(file)
    #  Inputs:
    #    file = filename
    #  Outputs:
    #    s = string
    
    df = pd.read_csv(fname, delimiter='\t', header=None)
    df[0] = df[0] - 1
    df[1] = df[1] - 1
    
    return df

def read_network(network_directory, layers, nfeatures):
    tic = time.perf_counter()
    cur_layer = 0
    next_layer = nfeatures
    edge_dfs = []
    for k in range(layers):
        filename = f"{network_directory}/n{nfeatures}-l{k+1}.tsv"
        df = readTriples(filename)
        layer_size = df.iloc[:,1].max()+2

        df.iloc[:,0] += cur_layer
        df.iloc[:,1] += next_layer
        
        cur_layer = next_layer
        next_layer += layer_size
        edge_dfs.append(df)

    edge_df = pd.concat(edge_dfs)

    readLayerTime = time.perf_counter() - tic
    readLayerRate = len(df.index)/readLayerTime;

    print('[INFO] DNN neurons: %d, layers: %d, edges: %d' %(next_layer, layers, len(edge_df.index)))
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
        featureVectors = scipy.sparse.random(int(inputs), nfeatures, density=0.5).tocsr()

    return featureVectors