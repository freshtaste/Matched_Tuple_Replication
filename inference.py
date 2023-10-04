import numpy as np
import statsmodels.api as sm

class Inference(object):

    def __init__(self, Y, D, A, design, tuple_idx=None, tau=0):
        self.Y = Y
        self.D = D
        self.A = A
        self.design = design
        if (design == '8' or design == '8p' or design == '9' or design == '10') and tuple_idx is None:
            raise ValueError("tuple_idx is required for matched tuple design")
        else:
            self.tuple_idx = tuple_idx
        self.tau = tau

        self.tau11, self.tau10, self.theta1, self.theta2, self.theta12 = self.estimator()
        #self.phi_tau1, self.phi_tau0, self.phi_theta = self.inference()

    def estimator(self):
        Y, D, A = self.Y, self.D, self.A
        mu00 = np.mean(Y[(D==0) & (A==0)])
        mu01 = np.mean(Y[(D==0) & (A==1)])
        mu10 = np.mean(Y[(D==1) & (A==0)])
        mu11 = np.mean(Y[(D==1) & (A==1)])

        tau11 = mu11 - mu01
        tau10 = mu10 - mu00
        tau21 = mu11 - mu10
        tau20 = mu01 - mu00
        
        theta1 = (tau11 + tau10)/2
        theta2 = (tau21 + tau20)/2
        theta12 = (mu11 - mu01 - (mu10 - mu00))/2
        return tau11, tau10, theta1, theta2, theta12

    def inference(self):
        n, d = len(self.Y), 4
        v11 = np.array([0,-1,0,1])
        v10 = np.array([-1,0,1,0])
        vt1 = (v11 + v10)/2
        vt2 = (np.array([0,0,-1,1]) + np.array([-1,1,0,0]))/2
        vt12 = (np.array([0,-1,0,1]) - np.array([-1,0,1,0]))/2
        
        if self.design == '1' or self.design == '2':
            Y, D, A = self.Y, self.D, self.A
            sigma00 = np.var(Y[(D==0) & (A==0)])
            sigma01 = np.var(Y[(D==0) & (A==1)])
            sigma10 = np.var(Y[(D==1) & (A==0)])
            sigma11 = np.var(Y[(D==1) & (A==1)])
            V = np.diag([sigma00, sigma01, sigma10, sigma11])
            
        elif self.design == '8' or self.design == '8p':
            Y_s = self.Y[self.tuple_idx] # (0,0) (0,1) (1,0), (1,1)
            # estimate Gamma
            gamma = np.mean(Y_s, axis=0)
            # estimate sigma2
            sigma2 = np.var(Y_s, axis=0)
            # estimate rho_dd
            rho2 = np.mean(Y_s[::2]*Y_s[1::2], axis=0)
            # estimate rho_dd'
            R = Y_s.T @ Y_s/(n/d)
            if self.design == '8':
                rho = R - np.diag(np.diag(R)) + np.diag(rho2)
            else:
                rho = (Y_s[::2].T @ Y_s[1::2] + Y_s[1::2].T @ Y_s[::2])/(n/d)
            # compute V
            V1 = np.diag(sigma2) - (np.diag(rho2) - np.diag(gamma**2))
            V2 = (rho - gamma.reshape(-1,1) @ gamma.reshape(-1,1).T)/d
            V = V1 + V2
            
        elif self.design == '9' or self.design == '10':
            Y, D, A = self.Y, self.D, self.A
            s = len(set(self.tuple_idx))
            # compute intermediate variables
            mu = np.zeros((s, d))
            for i in range(s):
                mu[i] = np.array([np.mean(Y[(D==d) & (A==a) & (self.tuple_idx==i)]) for d in range(2) for a in range(2)])
            Ybar = np.mean(mu, axis=0)
            Y2 = np.array([np.mean(Y[(D==d) & (A==a)]**2) for d in range(2) for a in range(2)])
            # compute variance
            sigma2 = Y2 - np.mean(mu**2, axis=0)
        else:
            raise ValueError("Design is not valid.")
        
        # compute variance estimator
        if self.design == '8' or self.design == '8p' or self.design == '1' or self.design == '2':
            var_tau11 = v11.dot(V).dot(v11)
            var_tau10 = v10.dot(V).dot(v10)
            var_theta1 = vt1.dot(V).dot(vt1)
            var_theta2 = vt2.dot(V).dot(vt2)
            var_theta12 = vt12.dot(V).dot(vt12)
        elif self.design == '9' or self.design == '10':
            var_tau11 = v11.dot(np.diag(sigma2)).dot(v11) + np.mean((mu - Ybar).dot(v11)**2)/4
            var_tau10 = v10.dot(np.diag(sigma2)).dot(v10) + np.mean((mu - Ybar).dot(v10)**2)/4
            var_theta1 = vt1.dot(np.diag(sigma2)).dot(vt1) + np.mean((mu - Ybar).dot(vt1*2)**2)/16
            var_theta2 = vt2.dot(np.diag(sigma2)).dot(vt2) + np.mean((mu - Ybar).dot(vt2*2)**2)/16
            var_theta12 = vt12.dot(np.diag(sigma2)).dot(vt12) + np.mean((mu - Ybar).dot(vt12*2)**2)/16
        else:
            raise ValueError("Design is not valid.")
        
        #print(self.theta1, var_theta1, np.sqrt(var_theta1/(n/d)), np.abs(self.theta1)/np.sqrt(var_theta1/(n/d)), self.tuple_idx)
        # compute reject probability
        phi_tau11 = 1 if np.abs(self.tau11)/np.sqrt(var_tau11/(n/d)) > 1.96 else 0
        phi_tau10 = 1 if np.abs(self.tau10)/np.sqrt(var_tau10/(n/d)) > 1.96 else 0
        phi_theta1 = 1 if np.abs(self.theta1)/np.sqrt(var_theta1/(n/d)) > 1.96 else 0
        phi_theta2 = 1 if np.abs(self.theta2)/np.sqrt(var_theta2/(n/d)) > 1.96 else 0
        phi_theta12 = 1 if np.abs(self.theta12)/np.sqrt(var_theta12/(n/d)) > 1.96 else 0
        self.var_tau11 = var_tau11
        self.var_tau10 = var_tau10
        self.var_theta1 = var_theta1
        self.var_theta2 = var_theta2
        self.var_theta12 = var_theta12
        self.se_tau10 = np.sqrt(self.var_tau10/(n/d))
        #phi_tau1 = 1 if np.abs(self.tau1-self.tau)/np.sqrt(var_tau1/(n/d)) <= 1.96 else 0
        #phi_tau0 = 1 if np.abs(self.tau0-self.tau)/np.sqrt(var_tau0/(n/d)) <= 1.96 else 0
        #phi_theta = 1 if np.abs(self.theta-self.tau)/np.sqrt(var_theta/(n/d)) <= 1.96 else 0
        return phi_tau11, phi_tau10, phi_theta1, phi_theta2, phi_theta12
    
    
class Inference2(Inference):
    
    def __init__(self, Y, D, A, cluster, tuple_idx=None, tau=0):
        self.Y = Y
        self.D = D
        self.A = A
        self.cluster = cluster
        self.tuple_idx = tuple_idx
        self.tau = tau
        self.tau11, self.tau10, self.theta1, self.theta2, self.theta12 = self.estimator()
        
    
    def inference(self, method):
        n, d = len(self.Y), 4
        v10 = np.array([-1,0,1,0])
        if method == 'mp':
            Y_s = self.Y[self.tuple_idx] # (0,0) (0,1) (1,0), (1,1)
            # estimate Gamma
            gamma = np.mean(Y_s, axis=0)
            # estimate sigma2
            sigma2 = np.var(Y_s, axis=0)
            # estimate rho_dd
            rho2 = np.mean(Y_s[::2]*Y_s[1::2], axis=0)
            # estimate rho_dd'
            R = Y_s.T @ Y_s/(n/d)
            rho = R - np.diag(np.diag(R)) + np.diag(rho2)
            # compute V
            V1 = np.diag(sigma2) - (np.diag(rho2) - np.diag(gamma**2))
            V2 = (rho - gamma.reshape(-1,1) @ gamma.reshape(-1,1).T)/d
            V = V1 + V2
            self.var_tau10 = v10.dot(V).dot(v10)
            self.se_tau10 = np.sqrt(self.var_tau10/(n/d))
            #print(V1)
            #print(V2)
            #print(Y_s[:10])
            #print(sigma2, '\n', rho2,'\n', gamma**2)
            #print(self.se_tau10)
        else:
            I11, I10, I01 = np.zeros(n), np.zeros(n), np.zeros(n)
            I11[(self.D==1) & (self.A==1)] = 1
            I10[(self.D==1) & (self.A==0)] = 1
            I01[(self.D==0) & (self.A==1)] = 1
            regressor = np.ones((n,4))
            regressor[:,1] = I10
            regressor[:,2] = I11
            regressor[:,3] = I01
            model = sm.OLS(self.Y,regressor)
            if method == 'robust':
                results = model.fit(cov_type='HC0')
                self.se_tau10 = results.bse[1]
            elif method == 'clustered':
                results = model.fit(cov_type='cluster', cov_kwds={'groups':self.cluster})
                self.se_tau10 = results.bse[1]
            else:
                raise ValueError("Inference method is not valid.")
        #print(self.se_tau10)
        if self.tau <= self.tau10 + 1.96*self.se_tau10 and self.tau >= self.tau10 - 1.96*self.se_tau10:
            cover = 1
        else:
            cover = 0
        # length of CI
        return cover