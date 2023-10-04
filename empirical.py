import numpy as np

class Inference(object):
    
    def __init__(self, Y, Yr):
        self.Y = Y # (0, 0, 1, 2)
        self.Yr = Yr
        self.tau1, self.tau2, self.tau12 = self.estimator()
        self.se_tau1, self.se_tau2, self.se_tau12 = self.inference()
        #if np.isnan(self.tau1) or np.isnan(self.tau2) or \
        #    np.isnan(self.se_tau1) or np.isnan(self.se_tau2):
        #        self.tau1, self.tau2, self.tau12 = self.estimator_nan()
        #        self.se_tau1, self.se_tau2, self.se_tau12 = self.inference_nan()

    def estimator(self):
        mu0 = np.mean(self.Y[:,:2])
        mu1 = np.mean(self.Y[:,2])
        mu2 = np.mean(self.Y[:,3])
        tau1 = mu1 - mu0
        tau2 = mu2 - mu0
        tau12 = mu2 - mu1
        return tau1, tau2, tau12

    def inference(self):
        v1 = np.array([-1/2,-1/2,1,0])
        v2 = np.array([-1/2,-1/2,0,1])
        v3 = np.array([0,0,-1,1])
        
        Y_s = self.Y # (0, 0, 1, 2)
        Y_r = self.Yr # with repeats
        n, d = len(Y_s)*4, 4
        # estimate Gamma
        gamma = np.mean(Y_s, axis=0)
        # estimate sigma2
        sigma2 = np.var(Y_s, axis=0)
        # estimate rho_dd
        if len(Y_r[::2]) != len(Y_r[1::2]):
            Y_even = np.concatenate([Y_r[1::2],Y_r[-2].reshape(1,-1)],axis=0)
        else:
            Y_even = Y_r[1::2]
        rho2 = np.mean(Y_r[::2]*Y_even, axis=0)
        # estimate rho_dd'
        R = Y_s.T @ Y_s/(n/d)
        rho = R - np.diag(np.diag(R)) + np.diag(rho2)
        #rho = (Y_r[::2].T @ Y_even + Y_even.T @ Y_r[::2])/(len(Y_r[::2])*2)
        # compute V
        V1 = np.diag(sigma2) - (np.diag(rho2) - np.diag(gamma**2))
        V2 = (rho - gamma.reshape(-1,1) @ gamma.reshape(-1,1).T)/d
        V = V1 + V2
        se_tau1 = np.sqrt(v1.dot(V).dot(v1)/(n/d))
        se_tau2 = np.sqrt(v2.dot(V).dot(v2)/(n/d))
        se_tau12 = np.sqrt(v3.dot(V).dot(v3)/(n/d))
        return se_tau1, se_tau2, se_tau12
    
    def estimator_nan(self):
        mu0 = np.nanmean(self.Y[:,:2])
        mu1 = np.nanmean(self.Y[:,2])
        mu2 = np.nanmean(self.Y[:,3])
        tau1 = mu1 - mu0
        tau2 = mu2 - mu0
        tau12 = mu2 - mu1
        return tau1, tau2, tau12
    
    def inference_nan(self):
        v1 = np.array([-1/2,-1/2,1,0])
        v2 = np.array([-1/2,-1/2,0,1])
        v3 = np.array([0,0,-1,1])
        
        Y_s = self.Y # (0, 0, 1, 2)
        Y_r = self.Yr # with repeats
        n, d = len(Y_s)*4, 4
        # estimate Gamma
        gamma = np.nanmean(Y_s, axis=0)
        # estimate sigma2
        sigma2 = np.nanvar(Y_s, axis=0)
        # estimate rho_dd
        if len(Y_r[::2]) != len(Y_r[1::2]):
            Y_even = np.concatenate([Y_r[1::2],Y_r[-2].reshape(1,-1)],axis=0)
        else:
            Y_even = Y_r[1::2]
        rho2 = np.nanmean(Y_r[::2]*Y_even, axis=0)
        # estimate rho_dd'
        R = np.array([[np.nansum(Y_s[i]*Y_s[j]) for i in range(4)] for j in range(4)])/(n/d)
        rho = R - np.diag(np.diag(R)) + np.diag(rho2)
        #rho = (Y_r[::2].T @ Y_even + Y_even.T @ Y_r[::2])/(len(Y_r[::2])*2)
        # compute V
        V1 = np.diag(sigma2) - (np.diag(rho2) - np.diag(gamma**2))
        V2 = (rho - gamma.reshape(-1,1) @ gamma.reshape(-1,1).T)/d
        V = V1 + V2
        se_tau1 = np.sqrt(v1.dot(V).dot(v1)/(n/d))
        se_tau2 = np.sqrt(v2.dot(V).dot(v2)/(n/d))
        se_tau12 = np.sqrt(v3.dot(V).dot(v3)/(n/d))
        return se_tau1, se_tau2, se_tau12