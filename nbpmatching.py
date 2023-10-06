import pandas as pd
import numpy as np
import os
#os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Resources"
from rpy2 import robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()


def nbpmatching(X):
    """
    Input a covariate matrix (n by k) and output an index matrix (n/2 by 2),
    whose first column represents the indices of group 1, second column
    represents the indieces of group 2
    """
    df = pd.DataFrame(X)
    idx = np.arange(len(X))
    df.insert(0,'id', idx)
    
    nbpmatch = robjects.r('''
        nbpmatch <- function(df) {
            library("nbpMatching")
            set.seed(123)
            df.dist <- gendistance(df, idcol=1)
            df.mdm <- distancematrix(df.dist)
            df.match <- nonbimatch(df.mdm)
            df.match[["halves"]]
            arr = c(as.numeric(unlist(df.match[["halves"]]["Group1.ID"])), 
                    as.numeric(unlist(df.match[["halves"]]["Group2.ID"])))
            matrix(arr, nrow = length(arr)/2, ncol = 2)
        }
        ''')
    nbpmatch2 = robjects.r('''
        nbpmatch <- function(df) {
            library("nbpMatching")
            df.1 <- runner(df, idcol=1)
            arr = c(as.numeric(unlist(df.1$matches$halves["Group1.Row"])), 
            as.numeric(unlist(df.1$matches$halves["Group2.Row"])))
            matrix(arr, nrow = length(arr)/2, ncol = 2)
        }
        ''')
    res = nbpmatch(df).astype(int)
    return res

def match_tuple(X, num_factor):
    """
    Match for k number of factors, i.e. group of size 2^k. Return the indices
    of m by group_size. 
    """
    n, k = X.shape
    group_size = 2**k
    #out = np.zeros((int(n/group_size), group_size))
    indices = []
    for f in range(num_factor):
        matched_idx = nbpmatching(X)
        #matched_idx = matched_idx - 1 # comment out if save error message "IndexError: index 1280 is out of bounds for axis 0 with size 1280"
        X = np.mean(X[matched_idx], axis=1)
        indices.append(matched_idx)
    real_idx = indices[0]
    for i in range(1,num_factor):
        idxi = indices[i]
        real_idx = np.concatenate((real_idx[idxi[:,0]], real_idx[idxi[:,1]]), axis=1)
    
    return real_idx

