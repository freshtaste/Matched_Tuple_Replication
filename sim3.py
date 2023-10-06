import numpy as np
import pandas as pd
from multiple_factor import DGP2, Inferece2
from joblib import Parallel, delayed
import multiprocessing
import statsmodels.api as sm
from nbpmatching import match_tuple
from scipy.stats import chi2
import pickle

np.random.seed(123)
# load covariates data
data = pd.read_csv("FactorialData/educationData2008.csv")
cols = ['Total']
cols += list(data.iloc[:,26:32].columns)
cols += list(data.iloc[:,34:36].columns)
cols += ['teachers']
covariates = data[cols].to_numpy()
covariates = covariates/np.std(covariates,axis=0)
covariates = covariates - np.mean(covariates,axis=0)
# add a few noise to break the tie for S4
covariates = covariates + 1e-5*np.random.normal(size=covariates.shape)
model = sm.OLS(covariates[:,-1], -covariates[:,:-1])
result = model.fit()
beta = result.params
residuals = result.resid
#print("Variance of Y0 vs epsilon",np.var(covariates[:,-1]), np.var(residuals))

class DGP3(DGP2):
    
    def __init__(self, num_factor, Xdim, num_sample, X, tau=0, match_more=False, design='MT'):
        self.total = X
        self.covariates = X[:,:-1]
        self.tuple_size = 2**num_factor
        self.num_factor = num_factor
        if num_sample%(self.tuple_size*2) == 0:
            self.n = num_sample
        else:
            raise ValueError("Number of sample needs to be 2^K*n.")
        self.tau = tau
        self.Xdim = Xdim
        self.match_more = match_more
        self.design = design
        if match_more:
            if self.design != 'MT':
                raise ValueError("match_more is true only under MT")
        self.tuple_idx = None
        self.all_treatments = self.get_treatment_combination()
        self.X = self.generate_X()
        self.D = self.generate_D()
        self.Y = self.generate_Y()
        if self.match_more:
            self.tuple_idx = self.tuple_idx.reshape(-1,self.tuple_size)
        
    def generate_X(self):
        idx = np.random.choice(len(self.total), self.n, replace=False)
        total = self.total[idx]
        self.Xtotal = total[:,:-1]
        X = total[:,:self.Xdim]
        self.Y0 = total[:,-1]
        return X
    
    def generate_D(self):
        if self.design == 'MT':
            self.tuple_idx = self.get_tuple_idx()
            df = pd.DataFrame(self.tuple_idx)
            idx = df.apply(lambda x:np.random.shuffle(x) or x, axis=1).to_numpy()
            D = np.zeros((self.n, self.num_factor))
            for c in range(idx.shape[1]):
                D[idx[:,c]] = np.array([np.array(self.all_treatments[c])]*int(self.n/len(self.all_treatments)))
        elif self.design == 'C':
            D = np.array(self.all_treatments*int(self.n/len(self.all_treatments)))
        elif self.design == 'S4':
            self.tuple_idx = np.zeros(self.n)
            D = np.zeros((self.n, self.num_factor))
            #X = (self.X - .5).dot(np.linspace(1,2,self.Xdim))
            X = self.X[:,np.random.choice(self.X.shape[1])]
            idx_s1, idx_s2 = X <= np.quantile(X, .25), (X <= np.median(X)) & (np.quantile(X, .25) < X)
            idx_s3, idx_s4 = (X <= np.quantile(X, .75)) & (np.median(X) < X), np.quantile(X, .75) < X
            D[idx_s1] = np.array(self.all_treatments*int(self.n/len(self.all_treatments)/4))
            D[idx_s2] = np.array(self.all_treatments*int(self.n/len(self.all_treatments)/4))
            D[idx_s3] = np.array(self.all_treatments*int(self.n/len(self.all_treatments)/4))
            D[idx_s4] = np.array(self.all_treatments*int(self.n/len(self.all_treatments)/4))
            self.tuple_idx[idx_s2] = 1
            self.tuple_idx[idx_s3] = 2
            self.tuple_idx[idx_s4] = 3
        elif self.design == 'RE':
            a = chi2.ppf(.01**(1/self.num_factor), self.Xdim)
            num_interaction = self.num_factor*(self.num_factor-1)/2
            if num_interaction == 0:
                b = 0
            else:
                b = chi2.ppf(.01**(1/num_interaction), self.Xdim)
            Mf_max = 100
            Mf_max_int = 100
            D = np.array(self.all_treatments*int(self.n/len(self.all_treatments)))
            while Mf_max > a or Mf_max_int > b:
                idx = np.random.permutation(self.n)
                D = D[idx]
                #taux = np.array([np.mean(self.X[D[:,f]==1] - self.X[D[:,f]==0], axis=0) for f in range(self.num_factor)])
                Mf_max = 0
                # compute maximum imbalance in main effects
                for f in range(self.num_factor):
                    x_diff = np.mean(self.X[D[:,f]==1] - self.X[D[:,f]==0], axis=0)
                    Mf = x_diff.dot(x_diff)*1*self.n/4
                    if Mf > Mf_max:
                        Mf_max = Mf
                Mf_max_int = 0
                # compute maximum imbalance in interaction effects
                for f1 in range(self.num_factor):
                    for f2 in range(f1+1, self.num_factor):
                        x_diff = np.mean(self.X[D[:,f1]==D[:,f2]] - self.X[D[:,f1]!=D[:,f2]], axis=0)
                        Mf_int = x_diff.dot(x_diff)*1*self.n/4
                        if Mf_int > Mf_max_int:
                            Mf_max_int = Mf_int
        elif self.design == 'MP-B':
            self.tuple_idx = match_tuple(self.X, 1)
            df = pd.DataFrame(self.tuple_idx)
            idx = df.apply(lambda x:np.random.shuffle(x) or x, axis=1).to_numpy()
            D = np.zeros((self.n, self.num_factor))
            D[idx[:,1],0] = 1
            D[:,1:] = np.random.choice([0,1], size=(self.n, self.num_factor-1))
        else:
            raise ValueError("Design is not valid.")
        return D

    def generate_Y(self):
        eps = np.random.normal(0, np.sqrt(0.1), size=self.n)
        if self.D.shape[1] > 1:
            gamma = 2*self.D[:,1] - 1
            #gamma = 1
            Y = gamma*self.Xtotal.dot(beta) \
                + (np.mean(self.D[:,1:],axis=1) + self.D[:,0])*self.tau + eps
        else:
            gamma = 1
            Y = gamma*self.Xtotal.dot(beta) \
                + self.D[:,0]*self.tau + eps
        return Y
    
    
def reject_prob(X, num_factor, Xdim, sample_size, tau=0, ntrials=1000, more=False, design='MT'):
    phi_tau = np.zeros(ntrials)
    for i in range(ntrials):
        dgp = DGP3(num_factor, Xdim, sample_size, X, tau, more, design)
        Y, D, tuple_idx = dgp.Y, dgp.D, dgp.tuple_idx
        inf = Inferece2(Y, D, tuple_idx, design)
        phi_tau[i] = inf.phi_tau
    return np.mean(phi_tau)

def risk(X, num_factor, Xdim, sample_size, tau=0, ntrials=1000, more=False, design='MT'):
    mse = np.zeros(ntrials)
    for i in range(ntrials):
        dgp = DGP3(num_factor, Xdim, sample_size, X, tau, more, design)
        Y, D, tuple_idx = dgp.Y, dgp.D, dgp.tuple_idx
        ate = np.mean(Y[D[:,0]==1]) - np.mean(Y[D[:,0]==0])
        mse[i] = (ate - tau)**2
    return np.mean(mse)

def reject_prob_parrell(X, num_factor, Xdim, sample_size, tau=0, ntrials=1000, more=False, design='MT'):
    if design == 'MT2':
        more = True
        design = 'MT'
    def process(qk):
        np.random.seed(123 + qk + num_factor*10 + Xdim)
        dgp = DGP3(num_factor, Xdim, sample_size, X, tau, more, design)
        Y, D, tuple_idx = dgp.Y, dgp.D, dgp.tuple_idx
        inf = Inferece2(Y, D, tuple_idx, design)
        return inf.phi_tau
    num_cores = multiprocessing.cpu_count() - 1
    ret = Parallel(n_jobs=num_cores)(delayed(process)(i) for i in range(ntrials))
    return np.mean(ret)

def risk_parrell(X, num_factor, Xdim, sample_size, tau=0, ntrials=1000, more=False, design='MT'):
    if design == 'MT2':
        more = True
        design = 'MT'
    def process(qk):
        np.random.seed(123 + qk + num_factor*10 + Xdim)
        dgp = DGP3(num_factor, Xdim, sample_size, X, tau, more, design)
        Y, D, tuple_idx = dgp.Y, dgp.D, dgp.tuple_idx
        ate = np.mean(Y[D[:,0]==1]) - np.mean(Y[D[:,0]==0])
        return (ate - tau)**2
    num_cores = multiprocessing.cpu_count() - 1
    ret = Parallel(n_jobs=num_cores)(delayed(process)(i) for i in range(ntrials))
    return np.mean(ret)


def get_saved_results():
    # set random seed
    np.random.seed(123)
    model_specs = [(1,1),(2,5),(6,5)]
    taus_list = [np.linspace(0, .1, 24), np.linspace(0, .1, 24), np.linspace(0, .15, 24)]

    results = {}
    for i, (q,k) in enumerate(model_specs):
        result = {}
        def processInput(t):
            c = reject_prob_parrell(covariates, k, q, 1280, tau=t, ntrials=1000, more=False, design='C')
            s4 = reject_prob_parrell(covariates, k, q, 1280, tau=t, ntrials=1000, more=False, design='S4')
            with open("sim3_runtime.txt", "a") as f:
                print("start mt", file=f)
            mt = reject_prob_parrell(covariates, k, q, 1280, tau=t, ntrials=1000, more=False, design='MT')
            mt2 = reject_prob_parrell(covariates, k, q, 1280, tau=t, ntrials=1000, more=True, design='MT')
            with open("sim3_runtime.txt", "a") as f:
                print("finish mt", file=f)
            return (mt,mt2,c,s4)
        #ret = Parallel(n_jobs=num_cores)(delayed(processInput)(t) for t in taus_list[i])
        ret = []
        for t in taus_list[i]:
            with open("sim3_runtime.txt", "a") as f:
                print(t, (q,k), file=f)
            tmp = processInput(t)
            ret.append(tmp)
            with open("sim3_runtime.txt", "a") as f:
                print(tmp, file=f)
            
        #ret = [processInput(t) for t in taus_list[i]]
        result['MT'] = [r[0] for r in ret]
        result['MT2'] = [r[1] for r in ret]
        result['C'] = [r[2] for r in ret]
        result['S4'] = [r[3] for r in ret]
        results["q={},k={}".format(q,k)] = result
        print(q,k)
        with open("simu3.txt", "a") as f:
            print((q,k), file=f)
            print(result, file=f)

    with open('sim3_power_plots.pkl', 'wb') as f:
        pickle.dump(results, f)


def get_Figure1_and_2():
    
    get_saved_results()

    with open('sim3_power_plots.pkl', 'rb') as f:
        results = pickle.load(f)
        
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    fig, axs = plt.subplots(1, 3, figsize=(30,12))
    styles = ['-', 'dashdot', ':', 'dashed']
    designs = ['MT','MT2','C','S4']
    width = [3,2,1.5,1.5]
    model_specs = [(1,1),(2,5),(6,5)]
    taus_list = [np.linspace(0, .1, 24), np.linspace(0, .1, 24), np.linspace(0, .15, 24)]
    rights = [.1, .1, .1]

    for i, (q,k) in enumerate(model_specs):
        for j, design in enumerate(designs):
            p = results["q={},k={}".format(q,k)][design]
            x, y = taus_list[i], p
            y = savgol_filter(p, 7, 3)
            axs[i].plot(x, y, label=design, color='black', linewidth=width[j], linestyle=styles[j])
            axs[i].set_ylim(bottom=0)
            axs[i].set_xlim(left=0, right=rights[i])
            axs[i].set_yticks([0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0])
            axs[i].xaxis.set_tick_params(labelsize=20)
            axs[i].yaxis.set_tick_params(labelsize=20)
        axs[i].set_title(r'Dim$(X_i)={}, K={}$'.format(q,k), fontdict={'fontsize': 30, 'fontweight': 'medium'})
        axs[i].legend(prop={'size': 24})
        
    plt.savefig("Figure2.pdf")


    fig, axs = plt.subplots(1, 3, figsize=(30,12))
    styles = ['-', 'dashdot', ':', 'dashed']
    designs = ['MT','MT2','C','S4']
    width = [3,2,1.5,1.5]
    model_specs = [(1,1),(2,5),(6,5)]
    taus_list = [np.linspace(0, .1, 24), np.linspace(0, .1, 24), np.linspace(0, .15, 24)]
    rights = [.1, .1, .15]

    for i, (q,k) in enumerate(model_specs):
        for j, design in enumerate(designs):
            p = results["q={},k={}".format(q,k)][design]
            x, y = taus_list[i], p
            y = savgol_filter(p, 7, 3)
            axs[i].plot(x, y, label=design, color='black', linewidth=width[j], linestyle=styles[j])
            axs[i].set_ylim(bottom=0)
            axs[i].set_xlim(left=0, right=rights[i])
            axs[i].set_yticks([0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0])
            axs[i].xaxis.set_tick_params(labelsize=20)
            axs[i].yaxis.set_tick_params(labelsize=20)
        axs[i].set_title(r'Dim$(X_i)={}, K={}$'.format(q,k), fontdict={'fontsize': 30, 'fontweight': 'medium'})
        axs[i].legend(prop={'size': 24})
        
    plt.savefig("Figure1.pdf")