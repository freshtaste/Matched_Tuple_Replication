import random
import numpy as np
import pandas as pd
from scipy.stats import chi2

class DGP(object):

    def __init__(self, model, design, num_sample, tau=0):
        self.model = model
        self.design = design
        if num_sample%4 == 0:
            self.n = num_sample
        else:
            raise ValueError("Number of sample needs to be 4*n.")
        self.tuple_idx = None # for design=8
        self.tau = tau

        self.X = self.generate_X()
        self.D, self.A = self.generate_DA()
        _, self.Y = self.generate_Y()
        # get cluster indictors
        if design == '8' or design == '8p':
            tmp = np.arange(int(self.n/4))
            cluster = np.zeros((self.n,))
            for i in range(4):
                cluster[self.tuple_idx[:,i]] = tmp
            self.cluster = cluster.reshape(-1,)

    def generate_X(self):
        #if self.model == '5':
        X = np.random.normal(0,1,self.n)
        #else:
        #    X = np.random.uniform(0,1,self.n)
        return X

    def generate_DA(self):
        n, X, design = self.n, self.X, self.design
        if design == '1':
            D = np.random.choice([0,1],size=n,p=[.5,.5])
            A = np.random.choice([0,1],size=n,p=[.5,.5])
        elif design == '2':
            D, A = self.crd(n)
        elif design == '3' or design == '4':
            idx = np.argsort(X).reshape(-1,2)
            chosen_col = np.random.choice([0,1],size=idx.shape[0],p=[.5,.5])
            idx_treated = idx[np.arange(idx.shape[0]), chosen_col]
            D = np.zeros(n)
            D[idx_treated] = 1
            if design == '3':
                A = np.random.choice([0,1],size=n,p=[.5,.5])
            else:
                A = np.zeros(n)
                A[int(n/2):] = 1
                A = np.random.permutation(A)
        elif design == '5' or design == '6':
            idx = np.argsort(X).reshape(-1,2)
            chosen_col = np.random.choice([0,1],size=idx.shape[0],p=[.5,.5])
            idx_treated = idx[np.arange(idx.shape[0]), chosen_col]
            A = np.zeros(n)
            A[idx_treated] = 1
            if design == '5':
                D = np.random.choice([0,1],size=n,p=[.5,.5])
            else:
                D = np.zeros(n)
                D[int(n/2):] = 1
                D = np.random.permutation(D)
        elif design == '7':
            idx = np.argsort(X).reshape(-1,2)
            chosen_col_D = np.random.choice([0,1],size=idx.shape[0],p=[.5,.5])
            chosen_col_A = np.random.choice([0,1],size=idx.shape[0],p=[.5,.5])
            idx_treated_D = idx[np.arange(idx.shape[0]), chosen_col_D]
            idx_treated_A = idx[np.arange(idx.shape[0]), chosen_col_A]
            D = np.zeros(n)
            D[idx_treated_D] = 1
            A = np.zeros(n)
            A[idx_treated_A] = 1
        elif design == '8' or design == '8p':
            idx = np.argsort(X).reshape(-1,4)
            df = pd.DataFrame(idx)
            idx = df.apply(lambda x:np.random.shuffle(x) or x, axis=1).to_numpy()
            self.tuple_idx = idx
            D, A = np.zeros(n), np.zeros(n)
            # (0,0) (0,1) (1,0), (1,1)
            D[idx[:,2]] = 1
            D[idx[:,3]] = 1
            A[idx[:,1]] = 1
            A[idx[:,3]] = 1
        elif design == '9':
            D, A = np.zeros(n), np.zeros(n)
            idx_s1, idx_s2 = self.X <= np.median(self.X), self.X > np.median(self.X)
            D1, A1 = self.crd(int(n/2))
            D2, A2 = self.crd(int(n/2))
            D[idx_s1] = D1
            A[idx_s1] = A1
            D[idx_s2] = D2
            A[idx_s2] = A2
            self.tuple_idx = np.zeros(n)
            self.tuple_idx[idx_s2] = 1
        elif design == '10':
            D, A = np.zeros(n), np.zeros(n)
            idx_s1, idx_s2 = self.X <= np.quantile(self.X, .25), (self.X <= np.median(self.X)) & (np.quantile(self.X, .25) < self.X)
            idx_s3, idx_s4 = (self.X <= np.quantile(self.X, .75)) & (np.median(self.X) < self.X), np.quantile(self.X, .75) < self.X
            D1, A1 = self.crd(int(n/4))
            D2, A2 = self.crd(int(n/4))
            D3, A3 = self.crd(int(n/4))
            D4, A4 = self.crd(int(n/4))
            D[idx_s1] = D1
            A[idx_s1] = A1
            D[idx_s2] = D2
            A[idx_s2] = A2
            D[idx_s3] = D3
            A[idx_s3] = A3
            D[idx_s4] = D4
            A[idx_s4] = A4
            self.tuple_idx = np.zeros(n)
            self.tuple_idx[idx_s2] = 1
            self.tuple_idx[idx_s3] = 2
            self.tuple_idx[idx_s4] = 3
        elif design == '11':
            a = chi2.ppf(.01**(1/2), 1)
            b = chi2.ppf(.01, 1)
            dist = 100
            dist2 = 100
            while dist > a or dist2 > b:
                D, A = np.zeros(n), np.zeros(n)
                D[int(n/2):] = 1
                A[int(n/4):int(n/2)] = 1
                A[int(3*n/4):] = 1
                idx = np.random.permutation(n)
                D = D[idx]
                A = A[idx]
                
                covX = 1
                
                x_diff_A = np.mean(self.X[D==1] - self.X[D==0])
                Mf_A = x_diff_A*(x_diff_A)*1/covX*self.n/4
                x_diff_B = np.mean(self.X[A==1] - self.X[A==0])
                Mf_B = x_diff_B*(x_diff_B)*1/covX*self.n/4
                dist = max(Mf_A, Mf_B)
                #print(dist, a)
                
                x_diff_interaction = np.mean(self.X[D==A] - self.X[D!=A])
                dist2 = x_diff_interaction*(x_diff_interaction)*1/covX*self.n/4
        else:
            raise ValueError('Design is not valid.')
        return D, A

    def generate_Y(self):
        n, X, D, A, model = self.n, self.X, self.D, self.A, self.model
        Y = {'0,0':np.zeros(n),
            '0,1':np.zeros(n),
            '1,0':np.zeros(n),
            '1,1':np.zeros(n)}
        sigma = {'0,0':np.ones(n),
            '0,1':np.ones(n)*2,
            '1,0':np.ones(n)*2,
            '1,1':np.ones(n)*3}
        eps = np.random.normal(0, 1, size=n)
        #gamma11, gamma10, gamma01, gamma00 = 1, -1, 1, -1
        gamma11, gamma10, gamma01, gamma00 = 2, 1/2, 1, -2
        
        if model == '1':
            Y['0,1'] = X + self.tau/2
            Y['1,1'] = X + 2*self.tau
            Y['0,0'] = X + 0
            Y['1,0'] = X + self.tau
        elif model == '2':
            Y['0,1'] = X + (X**2 - 1)/3 + self.tau/2
            Y['1,1'] = X + (X**2 - 1)/3 + 2*self.tau
            Y['0,0'] = X + (X**2 - 1)/3 
            Y['1,0'] = X + (X**2 - 1)/3 + self.tau
        elif model == '3':
            Y['0,1'] = gamma01*X + (X**2 - 1)/3 + self.tau/2
            Y['1,1'] = gamma11*X + (X**2 - 1)/3 + 2*self.tau
            Y['0,0'] = gamma00*X + (X**2 - 1)/3
            Y['1,0'] = gamma10*X + (X**2 - 1)/3 + self.tau
        elif model == '4':
            Y['0,1'] = np.sin(gamma01*X) + self.tau/2
            Y['1,1'] = np.sin(gamma11*X) + 2*self.tau
            Y['0,0'] = np.sin(gamma00*X)
            Y['1,0'] = np.sin(gamma10*X) + self.tau
        elif model == '5':
            Y['1,1'] = np.sin(gamma11*X) + gamma11*X/10 + (X**2 - 1)/3 + 2*self.tau
            Y['1,0'] = np.sin(gamma10*X) + gamma10*X/10 + (X**2 - 1)/3 + self.tau
            Y['0,1'] = np.sin(gamma01*X) + gamma01*X/10 + (X**2 - 1)/3 + self.tau/2
            Y['0,0'] = np.sin(gamma00*X) + gamma00*X/10 + (X**2 - 1)/3
        elif model == '6':
            Y['0,1'] = gamma01*X + (X**2 - 1)/3 + self.tau/2
            Y['1,1'] = gamma11*X + (X**2 - 1)/3 + 2*self.tau
            Y['0,0'] = gamma00*X + (X**2 - 1)/3
            Y['1,0'] = gamma10*X + (X**2 - 1)/3 + self.tau
            sigma['0,1'] *= 2*np.sqrt(np.abs(X))
            sigma['1,1'] *= 3*np.sqrt(np.abs(X))
            sigma['0,0'] *= np.sqrt(np.abs(X))
            sigma['1,0'] *= 2*np.sqrt(np.abs(X))
        else:
            raise ValueError('Model is not valid.')

        for k in Y.keys():
            if model == '6':
                Y[k] += eps*sigma[k]
            else:
                Y[k] += eps
    
        Yobs = np.zeros(n)
        Yobs[(D==0) & (A==0)] = Y['0,0'][(D==0) & (A==0)]
        Yobs[(D==0) & (A==1)] = Y['0,1'][(D==0) & (A==1)]
        Yobs[(D==1) & (A==0)] = Y['1,0'][(D==1) & (A==0)]
        Yobs[(D==1) & (A==1)] = Y['1,1'][(D==1) & (A==1)]
        
        self.tau10 = np.mean(Y['1,0']) - np.mean(Y['0,0'])
        
        return Y, Yobs

    def crd(self, n):
        D, A = np.zeros(n), np.zeros(n)
        D[int(n/2):] = 1
        A[int(n/4):int(n/2)] = 1
        A[int(3*n/4):] = 1
        idx = np.random.permutation(n)
        D = D[idx]
        A = A[idx]
        return D, A
    
    def get_data(self):
        return self.Y, self.D, self.A


class DGP_Finite(DGP):
    
    def __init__(self, model, num_sample, design='8', tau=0):
        self.model = model
        self.design = design
        if num_sample%4 == 0:
            self.n = num_sample
        else:
            raise ValueError("Number of sample needs to be 4*n.")
        self.tuple_idx = None # for design=8
        self.tau = tau

        self.X = self.generate_X()
        print(self.X[:10])
        self.D, self.A = self.generate_DA()
        self.Y, Yobs = self.generate_Y()
        self.tau10 = np.mean(self.Y['1,0']) - np.mean(self.Y['0,0'])
        if design == '8' or design == '8p':
            # get cluster indictors
            tmp = np.arange(int(self.n/4))
            cluster = np.zeros((self.n,))
            for i in range(4):
                cluster[self.tuple_idx[:,i]] = tmp
            self.cluster = cluster.reshape(-1,)
        
    def get_data(self):
        D, A = self.generate_DA()
        Yobs = np.zeros(self.n)
        Yobs[(D==0) & (A==0)] = self.Y['0,0'][(D==0) & (A==0)]
        Yobs[(D==0) & (A==1)] = self.Y['0,1'][(D==0) & (A==1)]
        Yobs[(D==1) & (A==0)] = self.Y['1,0'][(D==1) & (A==0)]
        Yobs[(D==1) & (A==1)] = self.Y['1,1'][(D==1) & (A==1)]
        return Yobs, D, A
        

#if __name__ == '__main__':
#    dgp = DGP('5','10',1000)
#    print(dgp.Y[:5])
#    print(dgp.design)

#    dgp = DGP_Finite('5',1000)
#    Yobs, D, A = dgp.get_data()
#    print(Yobs[:5])
#    Yobs, D, A = dgp.get_data()
#    print(Yobs[:5])