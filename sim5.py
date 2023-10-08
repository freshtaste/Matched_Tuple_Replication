import pandas as pd
import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
import multiprocessing


def get_inputed_outcome(df, method='homo'):
    df = df.copy()
    if method == 'homo':
        df['Y0'] = df['Y'].copy()
        df['Y1'] = df['Y'].copy()
        df['Y2'] = df['Y'].copy()
    else:
        df['Y0'] = np.nan
        df['Y1'] = np.nan
        df['Y2'] = np.nan
        for i in range(len(df)):
            strata_i = df['strata'].iloc[i]
            cov_i = df['continuous_covariate'].iloc[i]
            treatment_i = df['treatment'].iloc[i]
            df['Y{}'.format(treatment_i)].iloc[i] = df['Y'].iloc[i]
            other_treatment = [0,1,2]
            other_treatment.remove(treatment_i)
            for t in other_treatment:
                df_search = df.loc[(df['strata']==strata_i) & (df['treatment']==t)]
                # find the row with column continuous_covariate closes to cov_i
                df_search['dist'] = (df_search['continuous_covariate'] - cov_i).abs()
                min_dist = df_search['dist'].min()
                df['Y{}'.format(t)].iloc[i] = df_search.loc[df_search['dist']==min_dist, 'Y'].iloc[0]

    return df[['Y0', 'Y1', 'Y2']]
            

class DGP4():
    def __init__(self, num_sample, Xdiscret, Xcontinuous, Ys):
        self.num_sample = num_sample
        self.Xdiscret = Xdiscret
        self.Xcontinuous = Xcontinuous
        self.Ys = Ys
        self.treatment = None
        self.tuple_idx = None
        self.tau1 = self.Ys[:,1].mean() - self.Ys[:,0].mean()
        self.tau2 = self.Ys[:,2].mean() - self.Ys[:,0].mean()
        self.draw_samples()
        
    def draw_samples(self):
        idx = np.random.choice(len(self.Ys), self.num_sample, replace=True)
        self.Ys = self.Ys[idx]
        self.Xdiscret = self.Xdiscret[idx]
        self.Xcontinuous = self.Xcontinuous[idx]
        
    def get_treatment(self):
        # sort units by Xdiscret first and then Xcontinuous
        idx = np.lexsort((self.Xcontinuous, self.Xdiscret))
        idx = idx.reshape(-1,3)
        df = pd.DataFrame(idx)
        idx = df.apply(lambda x:np.random.shuffle(x) or x, axis=1).to_numpy()
        self.tuple_idx = idx
        # get treatment
        self.treatment = np.zeros(self.num_sample)
        self.treatment[idx[:,0]] = 0
        self.treatment[idx[:,1]] = 1
        self.treatment[idx[:,2]] = 2
        # get tuple indicator
        self.cluster = np.zeros(self.num_sample)
        values = np.zeros(idx.shape)
        values[:,0] = np.arange(len(idx))
        values[:,1] = np.arange(len(idx))
        values[:,2] = np.arange(len(idx))
        self.cluster[idx] = values
        
    def get_Y(self):
        self.Y = np.zeros(self.num_sample)
        self.Y[self.treatment==0] = self.Ys[self.treatment==0,0]
        self.Y[self.treatment==1] = self.Ys[self.treatment==1,1]
        self.Y[self.treatment==2] = self.Ys[self.treatment==2,2]

    def super_pop_draw(self):
        self.draw_samples()
        self.get_treatment()
        self.get_Y()
        return self.Y, self.treatment, self.tuple_idx
    
    def finite_pop_draw(self):
        self.get_treatment()
        self.get_Y()
        return self.Y, self.treatment, self.tuple_idx
        

class Inference3():
    def __init__(self, Y, treatment, cluster, tuple_idx, tau1, tau2):
        self.Y = Y
        self.treatment = treatment
        self.cluster = cluster
        self.tuple_idx = tuple_idx
        self.tau1 = tau1
        self.tau2 = tau2
        self.estimate()
    
    def estimate(self):
        self.tau1_hat = self.Y[self.treatment==1].mean() - self.Y[self.treatment==0].mean()
        self.tau2_hat = self.Y[self.treatment==2].mean() - self.Y[self.treatment==0].mean()
        return self.tau1_hat, self.tau2_hat
    
    def inference(self, method):
        n, d = len(self.Y), 3
        if method == 'mp':
            v10 = np.array([-1,1,0])
            v20 = np.array([-1,0,1])
            Y_s = self.Y[self.tuple_idx] # 0, 1, 2
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
            self.var_tau20 = v20.dot(V).dot(v20)
            self.se_tau20 = np.sqrt(self.var_tau20/(n/d))
        else:
            I1, I2 = np.zeros(n), np.zeros(n)
            I1[self.treatment==1] = 1
            I2[self.treatment==2] = 1
            regressor = np.ones((n,3))
            regressor[:,1] = I1
            regressor[:,2] = I2
            
            model = sm.OLS(self.Y,regressor)
            if method == 'robust':
                results = model.fit(cov_type='HC0')
                self.se_tau10 = results.bse[1]
                self.se_tau20 = results.bse[2]
            elif method == 'clustered':
                results = model.fit(cov_type='cluster', cov_kwds={'groups':self.cluster})
                self.se_tau10 = results.bse[1]
                self.se_tau20 = results.bse[2]
            else:
                raise ValueError("Inference method is not valid.")
        
        if self.tau1 <= self.tau1_hat + 1.96*self.se_tau10 and self.tau1 >= self.tau1_hat - 1.96*self.se_tau10:
            cover1 = 1
        else:
            cover1 = 0
            
        if self.tau2 <= self.tau2_hat + 1.96*self.se_tau20 and self.tau2 >= self.tau2_hat - 1.96*self.se_tau20:
            cover2 = 1
        else:
            cover2 = 0
        return cover1, cover2
    
    
def cover_rate(df, sample_size=900, modelY='homo', population='super', ntrials=2000):
    cover1 = np.zeros((ntrials, 3))
    cover2 = np.zeros((ntrials, 3))
    cf_length1 = np.zeros((ntrials, 3))
    cf_length2 = np.zeros((ntrials, 3))
    Ys = get_inputed_outcome(df, modelY)
    for i in range(ntrials):
        dgp = DGP4(sample_size, df['strata'].values, df['continuous_covariate'].values, Ys.values)
        if population == 'super':
            dgp.super_pop_draw()
        else:
            dgp.finite_pop_draw()
        inf = Inference3(dgp.Y, dgp.treatment, dgp.cluster, dgp.tuple_idx, dgp.tau1, dgp.tau2)
        
        cover1[i,0], cover2[i,0] = inf.inference('mp')
        cf_length1[i,0] = inf.se_tau10*1.96*2
        cf_length2[i,0] = inf.se_tau20*1.96*2
        
        cover1[i,1], cover2[i,1] = inf.inference('robust')
        cf_length1[i,1] = inf.se_tau10*1.96*2
        cf_length2[i,1] = inf.se_tau20*1.96*2
        
        cover1[i,2], cover2[i,2] = inf.inference('clustered')
        cf_length1[i,2] = inf.se_tau10*1.96*2
        cf_length2[i,2] = inf.se_tau20*1.96*2

    return np.mean(cover1, axis=0), np.mean(cf_length1, axis=0), np.mean(cover2, axis=0), np.mean(cf_length2, axis=0)


def get_table12():
    # set random seed
    np.random.seed(123)
    ghana = 'GHA_2008_MGFERE_v01_M_Stata8/ReplicationDataGhanaJDE.dta'
    baseline = 'GHA_2008_MGFERE_v01_M_Stata8/GhanaBaselineSingle_la.dta'
    columns_needed = ['realfinalprofit', 'atreatcash', 'atreatequip', 'wave', 'male', 'groupnum',
                        'sheno', 'male_male', 'female_female', 'male_mixed', 'female_mixed',
                        'highcapture','highcapital', 'highgroup', 'mlowgroup']

    data_ghana = pd.read_stata(ghana)[columns_needed]
    data_baseline_cov = pd.read_stata(baseline)[['SHENO', 'income_4b']]
    data_baseline_cov['sheno'] = data_baseline_cov['SHENO'].astype(int)
    # merge data_baseline_cov to data_ghana
    data_ghana = pd.merge(data_ghana, data_baseline_cov, how='left', left_on='sheno', right_on='sheno')
    # transform object columes into float, fill string values with 0
    data_ghana['income_4b'] = pd.to_numeric(data_ghana['income_4b'], errors='coerce').fillna(0)

    df_wave6 = data_ghana[data_ghana.wave==6]
    df_wave6['strata'] = df_wave6.male_male*100000 + df_wave6.female_female*10000 + df_wave6.male_mixed*1000 \
            + df_wave6.female_mixed*100 + df_wave6.highcapture*10 + df_wave6.highcapital
    df_wave6['continuous_covariate'] = df_wave6['income_4b']
    treatment = np.zeros(len(df_wave6))
    treatment[df_wave6['atreatcash']==1] = 1
    treatment[df_wave6['atreatequip']==1] = 2
    df_wave6['treatment'] = treatment
    df_wave6['treatment'] = df_wave6['treatment'].astype(int)
    df_wave6['Y'] = df_wave6.realfinalprofit
    # fill nan with 0
    df_wave6['Y'] = df_wave6['Y'].fillna(0)
    
    # start simulation
    modelYs = ['homo','hetero']
    sample_sizes = [60, 120, 360, 750, 1200]
    populations = ['super', 'finite']
    
    qkm_pairs = [(q,k,m) for q in modelYs for k in sample_sizes for m in populations]
    
    def processInput(qkm):
        q, k, m = qkm
        np.random.seed(123 + modelYs.index(q)*100 + sample_sizes.index(k)*10 + populations.index(m))
        cover1, cf1, cover2, cf2 = cover_rate(df_wave6, k, q, m, ntrials=1000)
        return (q,k,m, cover1, cf1, cover2, cf2)
    
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in qkm_pairs)
    output = np.zeros((len(modelYs)*3,len(sample_sizes)*2))
    cf_output = np.zeros((len(modelYs)*3,len(sample_sizes)*2))
    for (q,k,m,cover1,cf1,cover2,cf2) in results:
        if m == 'super':
            i = modelYs.index(q)
            j = sample_sizes.index(k)
            output[i*3:i*3+3,j] = cover1
            cf_output[i*3:i*3+3,j] = cf1
        else:
            i = modelYs.index(q)
            j = sample_sizes.index(k)
            output[i*3:i*3+3,j+len(sample_sizes)] = cover1
            cf_output[i*3:i*3+3,j+len(sample_sizes)] = cf1
            
    # print out output and cf_output rows by rows to file
    with open("Table12.txt", "a") as f:
        sample_sizes_repeat = sample_sizes*2
        for j in range(len(sample_sizes)*2):
            if j < len(sample_sizes)*2 - 1:
                print("$3n={}$ & ".format(sample_sizes_repeat[j]), end = '', file=f)
            else:
                print("$3n={}$ \\\\".format(sample_sizes_repeat[j]), file=f)
        
        for i in range(len(modelYs)*3):
            for j in range(len(sample_sizes)*2):
                if j < len(sample_sizes)*2 - 1:
                    print("{:.3f} & ".format(output[i,j]), end = '', file=f)
                else:
                    print("{:.3f} \\\\".format(output[i,j]), file=f)
            for j in range(len(sample_sizes)*2):
                if j < len(sample_sizes)*2 - 1:
                    print("({:.3f}) & ".format(cf_output[i,j]), end = '', file=f)
                else:
                    print("({:.3f}) \\\\".format(cf_output[i,j]), file=f)
        print("\n", file=f)