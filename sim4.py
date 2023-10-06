from dgp import DGP_Finite, DGP
from inference import Inference2, Inference
import numpy as np
import pandas as pd


def cover_rate(sample_size=1000, modelY='1', ntrials=2000):
    cover = np.zeros((ntrials, 3))
    cf_length = np.zeros((ntrials, 3))
    for i in range(ntrials):
        dgp = DGP(modelY, '8', sample_size)
        Yobs, D, A = dgp.get_data()
        inf = Inference2(Yobs, D, A, dgp.cluster, dgp.tuple_idx, dgp.tau)
        cover[i,0] = inf.inference('mp')
        cf_length[i,0] = inf.se_tau10*1.96*2
        cover[i,1] = inf.inference('robust')
        cf_length[i,1] = inf.se_tau10*1.96*2
        cover[i,2] = inf.inference('clustered')
        cf_length[i,2] = inf.se_tau10*1.96*2
    return np.mean(cover, axis=0), np.mean(cf_length, axis=0)


def cover_rate_finite(sample_size=1000, modelY='1', ntrials=2000):
    cover = np.zeros((ntrials, 3))
    cf_length = np.zeros((ntrials, 3))
    dgp = DGP_Finite(modelY, sample_size)
    for i in range(ntrials):
        Yobs, D, A = dgp.get_data()
        inf = Inference2(Yobs, D, A, dgp.cluster, dgp.tuple_idx, dgp.tau10)
        cover[i,0] = inf.inference('mp')
        cf_length[i,0] = inf.se_tau10*1.96*2
        cover[i,1] = inf.inference('robust')
        cf_length[i,1] = inf.se_tau10*1.96*2
        cover[i,2] = inf.inference('clustered')
        cf_length[i,2] = inf.se_tau10*1.96*2
    return np.mean(cover, axis=0), np.mean(cf_length, axis=0)


from joblib import Parallel, delayed
import multiprocessing


def get_table9():
    # set random seed
    np.random.seed(123)
    modelYs = ['1','2','3','4','5','6']

    sample_sizes = [40, 80, 160, 480, 1000]

    qk_pairs = [(q,k) for q in modelYs for k in sample_sizes]
    def processInput(qk):
        q, k = qk
        np.random.seed(123 + int(q)*10 + sample_sizes.index(k))
        cover, cf = cover_rate(k, q)
        return (q,k,cover,cf)
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in qk_pairs)
    output = np.zeros((len(modelYs)*3,len(sample_sizes)))
    cf_output = np.zeros((len(modelYs)*3,len(sample_sizes)))
    for (q,k,cover,cf) in results:
        i = int(q)-1
        j = sample_sizes.index(k)
        output[i*3:i*3+3,j] = cover
        cf_output[i*3:i*3+3,j] = cf

    output = np.zeros((len(modelYs)*3*2,len(sample_sizes)))
    for (q,k,cover,cf) in results:
        i = int(q)-1
        j = sample_sizes.index(k)
        out = [cover[0], cf[0], cover[1], cf[1], cover[2], cf[2]]
        output[i*6:i*6+6,j] = out
    print(output)
    pd.DataFrame(output).to_csv("Table9_part1.csv")
    
    def processInput2(qk):
        q, k = qk
        np.random.seed(123 - int(q)*10 + sample_sizes.index(k))
        cover, cf = cover_rate_finite(k, q)
        return (q,k,cover,cf)
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(processInput2)(i) for i in qk_pairs)
    output = np.zeros((len(modelYs)*3,len(sample_sizes)))
    cf_output = np.zeros((len(modelYs)*3,len(sample_sizes)))
    for (q,k,cover,cf) in results:
        i = int(q)-1
        j = sample_sizes.index(k)
        output[i*3:i*3+3,j] = cover
        cf_output[i*3:i*3+3,j] = cf

    output = np.zeros((len(modelYs)*3*2,len(sample_sizes)))
    for (q,k,cover,cf) in results:
        i = int(q)-1
        j = sample_sizes.index(k)
        out = [cover[0], cf[0], cover[1], cf[1], cover[2], cf[2]]
        output[i*6:i*6+6,j] = out
    print(output)
    pd.DataFrame(output).to_csv("Table9_part2.csv")