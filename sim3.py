import numpy as np
from sim2 import reject_prob_parrell
import pickle
import statsmodels.api as sm
import pandas as pd


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

def get_saved_results():
    # set random seed
    np.random.seed(123)
    model_specs = [(1,1),(2,5),(6,5)]
    taus_list = [np.linspace(0, .1, 24), np.linspace(0, .1, 24), np.linspace(0, .1, 24)]

    results = {}
    for i, (q,k) in enumerate(model_specs):
        result = {}
        def processInput(t):
            mt = reject_prob_parrell(covariates, q, k, 1280, tau=t, ntrials=500, more=False, design='MT')
            mt2 = reject_prob_parrell(covariates, q, k, 1280, tau=t, ntrials=500, more=True, design='MT')
            #mt, mt2 = 0, 1
            c = reject_prob_parrell(covariates, q, k, 1280, tau=t, ntrials=500, more=False, design='C')
            s4 = reject_prob_parrell(covariates, q, k, 1280, tau=t, ntrials=500, more=False, design='S4')
            return (mt,mt2,c,s4)
        ret = [processInput(t) for t in taus_list[i]]
        result['MT'] = [r[0] for r in ret]
        result['MT2'] = [r[1] for r in ret]
        result['C'] = [r[2] for r in ret]
        result['S4'] = [r[3] for r in ret]
        results["q={},k={}".format(q,k)] = result
        print(q,k)
        with open("Data_for_Figure1_and_2.txt", "a") as f:
            print((q,k), file=f)
            print(result, file=f)

    with open('simulation5_power_plots_new.pkl', 'wb') as f:
        pickle.dump(results, f)


def get_Figure1_and_2():
    
    get_saved_results()

    with open('simulation5_power_plots_new2.pkl', 'rb') as f:
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