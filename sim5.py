import numpy as np
#from sim_real import DGP3, reject_prob_parrell, reject_prob
from multiple_factor import Inferece2
import pickle
from joblib import Parallel, delayed
import multiprocessing


model_specs = [(1,1),(2,5),(6,5)]
taus_list = [np.linspace(0, .1, 24), np.linspace(0, .1, 24), np.linspace(0, .1, 24)]
num_cores = 50
"""
results = {}
for i, (q,k) in enumerate(model_specs):
    result = {}
    def processInput(t):
        mt = reject_prob_parrell(q, k, 1280, tau=t, ntrials=8, more=False, design='MT')
        mt2 = reject_prob_parrell(q, k, 1280, tau=t, ntrials=8, more=True, design='MT')
        #mt, mt2 = 0, 1
        c = reject_prob_parrell(q, k, 1280, tau=t, ntrials=8, more=False, design='C')
        s4 = reject_prob_parrell(q, k, 1280, tau=t, ntrials=8, more=False, design='S4')
        return (mt,mt2,c,s4)
    #ret = Parallel(n_jobs=num_cores)(delayed(processInput)(t) for t in taus_list[i])
    ret = [processInput(t) for t in taus_list[i]]
    result['MT'] = [r[0] for r in ret]
    result['MT2'] = [r[1] for r in ret]
    result['C'] = [r[2] for r in ret]
    result['S4'] = [r[3] for r in ret]
    results["q={},k={}".format(q,k)] = result
    print(q,k)
    with open("simulation_5.txt", "a") as f:
        print((q,k), file=f)
        print(result, file=f)

with open('simulation5_power_plots_new.pkl', 'wb') as f:
    pickle.dump(results, f)
"""


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
    
plt.savefig("power2.pdf")



import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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
    
plt.savefig("power2-short.pdf")