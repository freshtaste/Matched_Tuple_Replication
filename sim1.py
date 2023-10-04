import numpy as np
from dgp import DGP
from inference import Inference
from joblib import Parallel, delayed


def reject_prob_parrell(n, modelY='1', modelDA='1', ate=0, ntrials=1000):
    def process(i):
        dgp = DGP(modelY,modelDA,n,tau=ate)
        inf = Inference(dgp.Y, dgp.D, dgp.A, modelDA, tuple_idx=dgp.tuple_idx, tau=dgp.tau)
        return inf.inference()
    num_cores = 8
    ret = Parallel(n_jobs=num_cores)(delayed(process)(i) for i in range(ntrials))
    phi_tau11s, phi_tau10s, phi_theta1s, phi_theta2s, phi_theta12s = np.zeros(ntrials), np.zeros(ntrials), np.zeros(ntrials), np.zeros(ntrials), np.zeros(ntrials)
    for i in range(ntrials):
        phi_tau11s[i] = ret[i][0]
        phi_tau10s[i] = ret[i][1]
        phi_theta1s[i] = ret[i][2]
        phi_theta2s[i] = ret[i][3]
        phi_theta12s[i] = ret[i][4]
    return phi_tau11s, phi_tau10s, phi_theta1s, phi_theta2s, phi_theta12s

def risk_parrell(n, modelY='1', modelDA='1', ate=0, ntrials=1000):
    def process(i):
        dgp = DGP(modelY,modelDA,n,tau=ate)
        inf = Inference(dgp.Y, dgp.D, dgp.A, modelDA, tuple_idx=dgp.tuple_idx, tau=dgp.tau)
        return inf.tau11, inf.tau10, inf.theta1, inf.theta2, inf.theta12
    num_cores = 8
    ret = Parallel(n_jobs=num_cores)(delayed(process)(i) for i in range(ntrials))
    tau11s, tau10s, theta1s, theta2s, theta12s = np.zeros(ntrials), np.zeros(ntrials), np.zeros(ntrials), np.zeros(ntrials), np.zeros(ntrials)
    for i in range(ntrials):
        tau11s[i] = ret[i][0]
        tau10s[i] = ret[i][1]
        theta1s[i] = ret[i][2]
        theta2s[i] = ret[i][3]
        theta12s[i] = ret[i][4]
    return tau11s, tau10s, theta1s, theta2s, theta12s


def get_table3():
    # set random seed
    np.random.seed(123)
    # report MSE
    for i in range(6):
        print("ModelY={} (MSE)".format(i+1))
        mse = np.zeros((11,5))
        for j in range(11):
            tau11s, tau10s, theta1s, theta2s, theta12s = risk_parrell(1000, modelY=str(i+1), modelDA=str(j+1), ntrials=2000)
            mse[j,0] = np.mean(theta1s**2)
            mse[j,1] = np.mean(theta2s**2)
            mse[j,2] = np.mean(theta12s**2)
            mse[j,3] = np.mean(tau11s**2)
            mse[j,4] = np.mean(tau10s**2)
        mse = mse/mse[7]
        mse = mse.T
        with open("Table3.txt", "a") as f:
            print("ModelY={}".format(i+1),file=f)
            for r in range(5):
                for k in range(11):
                    if k<10:
                        print("{:.3f} & ".format(mse[r,k]), end = '', file=f)
                    else:
                        print("{:.3f} \\\\".format(mse[r,k]), file=f)
                for k in range(11):
                    if k<10:
                        if k in [0, 1, 2, 7, 8, 9]:
                            print("{:.3f} & ".format(mse[r,k]), end = '', file=f)
                    else:
                        print("{:.3f} \\\\".format(mse[r,k]), file=f)
            print("\n", file=f)
                    

def get_table4():
    # set random seed
    np.random.seed(123)
    # report Reject Probability with ate=0
    for i in range(6):
        print("ModelY={} (report Reject Probability with ate=0)".format(i+1))
        prob = np.zeros((5,5))
        for j, d in enumerate(['1', '2', '8', '9', '10']):
            phi_tau11s, phi_tau10s, phi_theta1s, phi_theta2s, phi_theta12s = reject_prob_parrell(1000, modelY=str(i+1), modelDA=d, ntrials=2000)
            prob[j,0] = np.mean(phi_theta1s)
            prob[j,1] = np.mean(phi_theta2s)
            prob[j,2] = np.mean(phi_theta12s)
            prob[j,3] = np.mean(phi_tau11s)
            prob[j,4] = np.mean(phi_tau10s)
        prob = prob.T
        with open("Table4_part1.txt", "a") as f:
            print("ModelY={}".format(i+1),file=f)
            for r in range(5):
                for k in range(5):
                    if k<4:
                        print("{:.3f} & ".format(prob[r,k]), end = '', file=f)
                    else:
                        print("{:.3f} \\\\".format(prob[r,k]), file=f)
            print("\n", file=f)
            
    for i in range(6):
        print("ModelY={} (report Reject Probability with ate=0.05)".format(i+1))
        prob = np.zeros((5,5))
        for j, d in enumerate(['1', '2', '8', '9', '10']):
            phi_tau11s, phi_tau10s, phi_theta1s, phi_theta2s, phi_theta12s = reject_prob_parrell(1000, modelY=str(i+1), modelDA=d, ate=0.2, ntrials=2000)
            prob[j,0] = np.mean(phi_theta1s)
            prob[j,1] = np.mean(phi_theta2s)
            prob[j,2] = np.mean(phi_theta12s)
            prob[j,3] = np.mean(phi_tau11s)
            prob[j,4] = np.mean(phi_tau10s)
        prob = prob.T
        with open("Table4_part2.txt", "a") as f:
            print("ModelY={}".format(i+1),file=f)
            for r in range(5):
                for k in range(5):
                    if k<4:
                        print("{:.3f} & ".format(prob[r,k]), end = '', file=f)
                    else:
                        print("{:.3f} \\\\".format(prob[r,k]), file=f)