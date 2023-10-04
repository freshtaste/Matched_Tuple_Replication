import numpy as np
import pandas as pd
from math import log2
from scipy.stats import chi2
import itertools
from nbpmatching import match_tuple


class DGP2(object):
    
    def __init__(self, num_factor, num_sample, Xdim, tau=0, match_more=False, design='MT'):
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
        self.Xall = None
        self.all_treatments = self.get_treatment_combination()
        self.X = self.generate_X()
        self.D = self.generate_D()
        self.Y = self.generate_Y()
        if self.match_more:
            self.tuple_idx = self.tuple_idx.reshape(-1,self.tuple_size)
        
    def generate_X(self):
        self.Xall = np.random.normal(0,1,size=(self.n,10))
        X = self.Xall[:,:self.Xdim]
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
            X = self.X[:,0]
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
        n, X, D = self.n, self.Xall, self.D
        eps = np.random.normal(0, 1, size=n)
        if D.shape[1] > 1:
            gamma = 2*D[:,1] - 1
            Y = gamma*X.dot(np.linspace(1,.1,10)) \
                + (np.mean(D[:,1:],axis=1) + D[:,0])*self.tau + eps
        else:
            gamma = 1
            Y = gamma*X.dot(np.linspace(1,.1,10)) \
                + D[:,0]*self.tau + eps
        return Y
        
    def get_treatment_combination(self):
        lst = list(itertools.product([0, 1], repeat=self.num_factor))
        if self.match_more:
            return lst + lst
        return lst
    
    def get_tuple_idx(self):
        """
        Get a match_tuple of shape (-1, 2^(K+1)) and then transform it into 
        shape (-1, 2^K) in order to calculate variance estimator
        """
        tuple_idx = match_tuple(self.X, self.num_factor+1)
        if self.match_more:
            return tuple_idx
        return tuple_idx.reshape(-1,self.tuple_size)
        


class Inferece2(object):
    
    def __init__(self, Y, D, tuple_idx, design='MT'):
        self.Y = Y
        self.D = D
        self.n, self.tuple_size = self.D.shape[0], 2**(self.D.shape[1])
        self.num_factor = int(log2(self.tuple_size))
        self.tuple_idx = tuple_idx
        self.design = design
        self.phi_tau, self.phi_tau_p = None, None
        self.tau = self.estimator()
        self.inference()
        
    def estimator(self):
        Y, D = self.Y, self.D
        tau = np.mean(Y[D[:,0]==1]) - np.mean(Y[D[:,0]==0])
        return tau
    
    def get_reject(self, rho_type='classic'):
        n, d = len(self.Y), self.tuple_size
        Y_s = self.Y[self.tuple_idx] # (0,0) (0,1) (1,0), (1,1)
        # estimate Gamma
        gamma = np.mean(Y_s, axis=0)
        # estimate sigma2
        sigma2 = np.var(Y_s, axis=0)
        # estimate rho_dd
        rho2 = np.mean(Y_s[::2]*Y_s[1::2], axis=0)
        # estimate rho_dd'
        R = Y_s.T @ Y_s/(n/d)
        if rho_type == 'classic':
            rho = R - np.diag(np.diag(R)) + np.diag(rho2)
        else:
            rho = (Y_s[::2].T @ Y_s[1::2] + Y_s[1::2].T @ Y_s[::2])/(n/d)
        # compute V
        V1 = np.diag(sigma2) - (np.diag(rho2) - np.diag(gamma**2))
        V2 = (rho - gamma.reshape(-1,1) @ gamma.reshape(-1,1).T)/d
        V = V1 + V2
        # compute variance
        v = self.get_select_vector()
        var_tau = v.dot(V).dot(v)
        #print(self.tau, var_tau, np.sqrt(var_tau/(n/d)), np.abs(self.tau)/np.sqrt(var_tau/(n/d)), V, self.tuple_idx)
        # compute reject probability
        phi_tau = 1 if np.abs(self.tau)/np.sqrt(var_tau/(n/d)) > 1.96 else 0
        return phi_tau
    
    def inference_MT(self):
        return self.get_reject(rho_type='classic'), self.get_reject(rho_type='pairs-on-pairs')
    
    def inference(self):
        all_treatments = list(itertools.product([0, 1], repeat=self.num_factor))
        v = self.get_select_vector()
        if self.design == 'MT':
            self.phi_tau, self.phi_tau_p = self.inference_MT()
        elif self.design == 'C':
            V = np.diag([np.var(self.Y[np.prod(self.D==d,axis=1)==1]) for d in all_treatments])
            # compute variance
            var_tau = v.dot(V).dot(v)
            self.phi_tau = 1 if np.abs(self.tau)/np.sqrt(var_tau/(self.n/self.tuple_size)) > 1.96 else 0
        elif self.design == 'S4':
            num_strata = len(set(self.tuple_idx))
            num_treatment = len(all_treatments)
            sigma = np.zeros((num_strata,num_treatment))
            mu = np.zeros((num_strata,num_treatment))
            for i, d in enumerate(all_treatments):
                for s in range(num_strata):
                    Y_ds = self.Y[(np.prod(self.D==d,axis=1)==1) & (self.tuple_idx==s)]
                    #sigma[s, d] = np.var(Y_ds)
                    mu[s, i] = np.mean(Y_ds)
            #V1 = np.diag(np.mean(sigma, axis=0))
            #V2 = np.cov(mu.T)/num_treatment
            #V = V1 + V2
            Ybar = np.mean(mu, axis=0)
            Y2 = np.array([np.mean(self.Y[np.prod(self.D==d,axis=1)==1]**2) for d in all_treatments])
            sigma2 = Y2 - np.mean(mu**2, axis=0)
            var_tau = v.dot(np.diag(sigma2)).dot(v) + np.mean((mu - Ybar).dot(v)**2)/num_treatment
            #print(self.tau,var_tau,sigma2,v.dot(np.diag(sigma2)).dot(v), Y2,np.mean(mu**2, axis=0))
            self.phi_tau = 1 if np.abs(self.tau)/np.sqrt(var_tau/(self.n/self.tuple_size)) > 1.96 else 0
        else:
            raise ValueError("Design is not valid.")
        
        return None
    
    def get_select_vector(self):
        v = np.zeros(self.tuple_size)
        mid = int(self.tuple_size/2)
        v[mid:] = 1/mid
        v[:mid] = -1/mid
        return v