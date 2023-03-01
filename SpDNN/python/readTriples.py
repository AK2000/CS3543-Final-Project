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