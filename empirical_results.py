import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from empirical import Inference


# inference based on matched-tuples
def reg_MT(df, column):
    Y = df.realfinalprofit.to_numpy()
    if column == 1:
        idx = Y>=0
    elif column == 2:
        idx = (df.male==1)
    elif column == 3:
        idx =  (df.male==0)
    elif column == 4:
        idx = (df.male==0) & (df.highgroup==1)
    elif column == 5:
        idx = (df.male==0) & (df.mlowgroup==1)
    else:
        raise RuntimeError("Wrong column")
    Y = Y[idx]
    Y = Y.reshape(-1,4)
    inf = Inference(Y, Y)
    tstats = np.abs(inf.tau12)/inf.se_tau12
    pval = (1-norm.cdf(tstats))*2
    return [inf.tau1, inf.se_tau1, inf.tau2, inf.se_tau2, pval]

# replicate regression from Table 5 of GhanaJDE paper
def reg(df, column, dummies, fixed_effects=True, clustered_se=False):
    Y = df.realfinalprofit
    if fixed_effects:
        X = df[['atreatcash', 'atreatequip']+list(dummies.columns)[1:]]
    else:
        X = df[['atreatcash', 'atreatequip']]
    if column == 1:
        idx = Y>=0
    elif column == 2:
        idx = (df.male==1)
    elif column == 3:
        idx =  (df.male==0)
    elif column == 4:
        idx = (df.male==0) & (df.highgroup==1)
    elif column == 5:
        idx = (df.male==0) & (df.mlowgroup==1)
    else:
        raise RuntimeError("Wrong column")
    Y = Y[idx]
    X = X[idx]
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    if clustered_se:
        results = model.fit(cov_type='cluster', cov_kwds={'groups':df.groupnum[idx]})
    else:
        results = model.fit(cov_type='HC0')
    #print(results.params[1:3])
    #print(results.HC0_se[1:3])
    r = np.zeros_like(results.params)
    r[1:3] = [1,-1]
    T_test = results.t_test(r)
    #print(T_test.pvalue)
    #print(results.params)
    #print(results.summary())
    if clustered_se:
        return results.params[1:3].values, results.bse[1:3].values, T_test.pvalue
    else:
        return results.params[1:3].values, results.HC0_se[1:3].values, T_test.pvalue


def print_results(results):
    stars = [norm.ppf(0.95), norm.ppf(0.975), norm.ppf(0.995)]
    pvals = [0.1, 0.05, 0.01]
    with open("Table7.txt", "a") as f:
        print("  (1)   (2)   (3)   (4)   (5)", file=f)
        for r in range(5):
            for i in range(5):
                if r==1 or r==3:
                    print(" & & ({:.2f})".format(results[i][r]), end=' ', file=f)
                elif r==4:
                    star = ''
                    for p in pvals:
                        if results[i][r] < p:
                            star += '*'
                    star = "^{" +star+"}" if len(star) > 0 else ''
                    print(" & & {:.3f}".format(results[i][r]), end=' ', file=f)
                else:
                    tstats = np.abs(results[i][r])/results[i][r+1]
                    star = ''
                    for s in stars:
                        if tstats > s:
                            star += '*'
                    star = "^{" +star+"}" if len(star) > 0 else ''
                    print(" & & {:.2f}".format(results[i][r]) + star, end=' ', file=f)
            print("\\\\", file=f)



def get_Table7():
    ghana = 'GHA_2008_MGFERE_v01_M_Stata8/ReplicationDataGhanaJDE.dta'
    data_ghana = pd.read_stata(ghana)
    
        # recovering large strata by baseline covariates and add a variable indicating treatment status {0,1,2}
    columns_needed = ['realfinalprofit', 'atreatcash', 'atreatequip', 'wave', 'male', 'groupnum',
                    'sheno', 'male_male', 'female_female', 'male_mixed', 'female_mixed',
                    'highcapture','highcapital', 'highgroup', 'mlowgroup']
    df_wave6 = data_ghana[data_ghana.wave==6][columns_needed]
    df_wave6['strata'] = df_wave6.male_male*100000 + df_wave6.female_female*10000 + df_wave6.male_mixed*1000 \
        + df_wave6.female_mixed*100 + df_wave6.highcapture*10 + df_wave6.highcapital
    treatment = np.zeros(len(df_wave6))
    treatment[df_wave6['atreatcash']==1] = 1
    treatment[df_wave6['atreatequip']==1] = 2
    df_wave6['treatment'] = treatment
    df_wave6 = df_wave6.sort_values(by=['strata','groupnum','treatment'], ascending=True)
    df_nan = df_wave6[np.isnan(df_wave6.realfinalprofit)]
    groups_nan = set(df_nan.groupnum)

    # drop 4 non-quadruplets groups and groups with missing values
    bad_groups = set([991,992,993,994] + list(groups_nan))
    df_wave6_quad = df_wave6[~df_wave6['groupnum'].isin(bad_groups)]

    # keep relavant variables and create dummy variables for group fixed effects
    dummies = pd.get_dummies(df_wave6_quad.groupnum)
    df_wave6_quad = pd.concat([df_wave6_quad, dummies], axis=1, join='inner')

        # regression with strata fixed effects
    #print("*************** With fixed effects ***************")
    results = []
    for i in range(5):
        taus, ses, pval = reg(df_wave6_quad, i+1, dummies, fixed_effects=True)
        results.append([taus[0], ses[0], taus[1], ses[1], pval])
        
    #print_results(results)

    # regression without strata fixed effects
    #print("*************** Without fixed effects ***************")
    results = []
    for i in range(5):
        taus, ses, pval = reg(df_wave6_quad, i+1, dummies, fixed_effects=False)
        results.append([taus[0], ses[0], taus[1], ses[1], pval])
        
    print_results(results)

    # regression without strata fixed effects under clustered standard error
    #print("*************** Without fixed effects under Clustered standard error ***************")
    results = []
    for i in range(5):
        taus, ses, pval = reg(df_wave6_quad, i+1, dummies, fixed_effects=False, clustered_se=True)
        results.append([taus[0], ses[0], taus[1], ses[1], pval])
        
    #print_results(results)

    # MT
    #print("*************** MT ***************")
    results = []
    for i in range(5):
        results.append(reg_MT(df_wave6_quad, i+1))
        
    print_results(results)


# replicate regression from Table 5 of GhanaJDE paper
def reg2(df, column, dummies, fixed_effects=True, clustered_se=False):
    Y = df.realfinalprofit
    if fixed_effects:
        X = df[['atreatcash', 'atreatequip']+list(dummies.columns)]
    else:
        X = df[['atreatcash', 'atreatequip']]
    if column == 1:
        idx = (1-np.isnan(Y))==1
    elif column == 2:
        idx = ((1-np.isnan(Y))==1) &(df.male==1)
    elif column == 3:
        idx = ((1-np.isnan(Y))==1) &(df.male==0)
    elif column == 4:
        idx = ((1-np.isnan(Y))==1) &(df.male==0) & (df.highgroup==1)
    elif column == 5:
        idx = ((1-np.isnan(Y))==1) &(df.male==0) & (df.mlowgroup==1)
    else:
        raise RuntimeError("Wrong column")
    Y = Y[idx]
    X = X[idx]
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    if clustered_se:
        results = model.fit(cov_type='cluster', cov_kwds={'groups':df.groupnum[idx]})
    else:
        results = model.fit(cov_type='HC0')
    #print(results.params[1:3])
    #print(results.HC0_se[1:3])
    r = np.zeros_like(results.params)
    r[1:3] = [1,-1]
    T_test = results.t_test(r)
    #print(T_test.pvalue)
    #print(results.params)
    #print(results.summary())
    if clustered_se:
        return results.params[1:3].values, results.bse[1:3].values, T_test.pvalue
    else:
        return results.params[1:3].values, results.HC0_se[1:3].values, T_test.pvalue
    
    
# inference based on matched-tuples
def reg_MT2(df, column):
    Y = df.realfinalprofit.to_numpy()
    if column == 1:
        # all
        idx = np.arange(len(Y))
    elif column == 2:
        idx = (df.male==1)
    elif column == 3:
        idx =  (df.male==0)
    elif column == 4:
        idx = (df.male==0) & (df.highgroup==1)
    elif column == 5:
        idx = (df.male==0) & (df.mlowgroup==1)
    else:
        raise RuntimeError("Wrong column")
    Y = Y[idx]
    Y = Y.reshape(-1,4)
    inf = Inference(Y, Y)
    tstats = np.abs(inf.tau12)/inf.se_tau12
    pval = (1-norm.cdf(tstats))*2
    return [inf.tau1, inf.se_tau1, inf.tau2, inf.se_tau2, pval]


def print_results2(results):
    stars = [norm.ppf(0.95), norm.ppf(0.975), norm.ppf(0.995)]
    pvals = [0.1, 0.05, 0.01]
    with open("Table11.txt", "a") as f:
        print("  (1)   (2)   (3)   (4)   (5)", file=f)
        for r in range(5):
            for i in range(5):
                if r==1 or r==3:
                    print(" & & ({:.2f})".format(results[i][r]), end=' ', file=f)
                elif r==4:
                    star = ''
                    for p in pvals:
                        if results[i][r] < p:
                            star += '*'
                    star = "^{" +star+"}" if len(star) > 0 else ''
                    print(" & & {:.3f}".format(results[i][r]), end=' ', file=f)
                else:
                    tstats = np.abs(results[i][r])/results[i][r+1]
                    star = ''
                    for s in stars:
                        if tstats > s:
                            star += '*'
                    star = "^{" +star+"}" if len(star) > 0 else ''
                    print(" & & {:.2f}".format(results[i][r]) + star, end=' ', file=f)
            print("\\\\", file=f)

def get_table11():
    # append wave7 data to wave1-6
    longterm = 'GHA_2008_MGFERE_v01_M_Stata8/r7tomerge.dta'
    baseline = 'GHA_2008_MGFERE_v01_M_Stata8/ReplicationDataGhanaJDE.dta'
    data_longterm = pd.read_stata(longterm)
    data_baseline = pd.read_stata(baseline)
    data_baseline['male'] = np.zeros(len(data_baseline))
    data_baseline['male'][data_baseline.gender=='male'] = 1
    data_all_waves = data_baseline.append(data_longterm, ignore_index=True)
    data_all_waves = data_all_waves.sort_values(by=['sheno','wave'], ascending=True)

    # fill in useful information/baseline covariates from wave1-6 to wave7
    # (we replicate the same stata commends from Table 5 in JDEreplicationfilesGhana.do
    baseline_columns = ['atreatcash', 'atreatequip', 'male', 'groupnum', 'male_male', 'female_female', 
                        'male_mixed', 'female_mixed', 'highcapture','highcapital','highgroup', 'mlowgroup']

    for i in range(1,len(data_all_waves)):
        if data_all_waves['sheno'].iloc[i] != data_all_waves['sheno'].iloc[i-1]:
            for col in baseline_columns:
                if np.isnan(data_all_waves[col].iloc[i-1]):
                    data_all_waves[col].iloc[i-1] = data_all_waves[col].iloc[i-2] if not np.isnan(data_all_waves[col].iloc[i-2]) \
                                                                        else data_all_waves[col].iloc[i-3]
                else:
                    raise RuntimeError("It should be nan.")
        if i == len(data_all_waves) - 1:
            for col in baseline_columns:
                data_all_waves[col].iloc[i] = data_all_waves[col].iloc[i-1] if not np.isnan(data_all_waves[col].iloc[i-1]) \
                                                                        else data_all_waves[col].iloc[i-2]
                                                                        
    # recovering large strata by baseline covariates and add a variable indicating treatment status {0,1,2}
    columns_needed = ['realfinalprofit', 'atreatcash', 'atreatequip', 'wave', 'male', 'groupnum',
                    'sheno', 'male_male', 'female_female', 'male_mixed', 'female_mixed',
                    'highcapture','highcapital', 'highgroup', 'mlowgroup']
    df_wave7 = data_all_waves[data_all_waves.wave==7][columns_needed]
    df_wave7['strata'] = df_wave7.male_male*100000 + df_wave7.female_female*10000 + df_wave7.male_mixed*1000 \
        + df_wave7.female_mixed*100 + df_wave7.highcapture*10 + df_wave7.highcapital
    treatment = np.zeros(len(df_wave7))
    treatment[df_wave7['atreatcash']==1] = 1
    treatment[df_wave7['atreatequip']==1] = 2
    df_wave7['treatment'] = treatment
    df_wave7 = df_wave7.sort_values(by=['strata','groupnum','treatment'], ascending=True)

    # keep relavant variables and create dummy variables for group fixed effects
    dummies = pd.get_dummies(df_wave7.groupnum)
    df_wave7 = pd.concat([df_wave7, dummies], axis=1, join='inner')
    
    bad_groups = [991,992,993,994]
    df_wave7_quad = df_wave7[~df_wave7['groupnum'].isin(bad_groups)]
    
    #print("*************** With fixed effects ***************")
    results = []
    for i in range(5):
        taus, ses, pval = reg2(df_wave7_quad, i+1, dummies, fixed_effects=True)
        results.append([taus[0], ses[0], taus[1], ses[1], pval])
        
    print_results2(results)

    # regression without strata fixed effects
    #print("*************** Without fixed effects ***************")
    results = []
    for i in range(5):
        taus, ses, pval = reg2(df_wave7_quad, i+1, dummies, fixed_effects=False)
        results.append([taus[0], ses[0], taus[1], ses[1], pval])
        
    print_results2(results)

    # regression without strata fixed effects under clustered standard error
    #print("*************** Without fixed effects under Clustered standard error ***************")
    results = []
    for i in range(5):
        taus, ses, pval = reg2(df_wave7_quad, i+1, dummies, fixed_effects=False, clustered_se=True)
        results.append([taus[0], ses[0], taus[1], ses[1], pval])
        
    #print_results2(results)

    # MT
    #print("*************** MT ***************")
    results = []
    for i in range(5):
        results.append(reg_MT2(df_wave7_quad, i+1))
        
    print_results2(results)