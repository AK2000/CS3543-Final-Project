import os
import pandas as pd
import scipy.sparse
import tqdm

name_base = "dataset/networks/random/{}"
max_layers = 120
nfeatures = 1024
for sparsity in [0.9, 0.5, 0.1, 0.05, 0.01]:
    folder = name_base.format(sparsity)
    os.makedirs(folder, exist_ok=True)
    print("Generating network: ", folder)
    for layer in tqdm.tqdm(range(1,max_layers+1)):
        weights = scipy.sparse.random(nfeatures, nfeatures, density=sparsity)
        weights = weights / np.sqrt(nfeatures)

        df = pd.DataFrame({
                0: weights.row,
                1: weights.col,
                2: weights.data
            })
        
        filename = f"{folder}/n{nfeatures}-l{layer}.tsv"
        df.to_csv(filename, sep="\t", header=False, index=False)