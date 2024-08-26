#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python
"""Model fit for competing Action Value Learning and Stimulus Value Learning in the cotext of Reinforcement Learning at the individual level"""

import numpy as np
import pandas as pd
import stan
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
from Madule import utils
import arviz as az
from scipy import stats 

def find_mode_point(data, num_bins=1000):

    # Using guissian kernel
    kde = stats.gaussian_kde(data)
    data_kde = kde.resample()[0]

    # Discretize the data into bins
    bins = np.linspace(min(data_kde), max(data_kde), num_bins)
    digitized = np.digitize(data_kde, bins)

    # Find the mode of the digitized data
    mode_bin = stats.mode(digitized)

    # Find the midpoint of the mode bin
    mode_point = (bins[mode_bin.mode - 1] + bins[mode_bin.mode]) / 2

    return mode_point


simNumber = 1
#parent_dir = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/simulation/' # main Folder
parent_dir = '/mnt/scratch/projects/7TPD/amin' # sractch folder
# List of subjects
subList = ['sub-004', 'sub-010', 'sub-012', 'sub-025', 'sub-026', 'sub-029', 'sub-030',
           'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042', 'sub-044', 
           'sub-045', 'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-059', 
           'sub-060', 'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071', 
           'sub-074', 'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 
           'sub-081', 'sub-082', 'sub-083', 'sub-085', 'sub-087', 'sub-088', 'sub-089', 
           'sub-090', 'sub-092', 'sub-108', 'sub-109']
# number of runs
n_runs = 10
# delcare a array for saving mean parameters, (run, subjs, conds, session, para)
subs_para_mean = np.zeros([n_runs, len(subList), 2, 2, 3])
# delcare a array for saving mode parameters
subs_para_mode = np.zeros([n_runs, len(subList), 2, 2, 3])
# loop over list of participants
for run in range(1, n_runs+1):
    print(run)
    for i, subName in enumerate(subList):
        for session in [1, 2]: 
            pickelDir = parent_dir + '/sim/' + str(simNumber) + '/' + subName + '/sess-' + str(session) + '_RL_condition_same_lr_run'+str(run)+'.pkl'
            """Loading the pickle file of model fit from the subject directory"""
            loadPkl = utils.load_pickle(load_path=pickelDir)
            fit = loadPkl['fit']   
            # mean of estimated paraemter  for Action condition
            subs_para_mean[run-1, i, 0, session-1] = np.array([fit['alpha'][0].mean(), fit['weightAct'][0].mean(), fit['sensitivity'][0].mean()])
            # mean of estimated paraemter  for Action condition
            subs_para_mean[run-1,i, 1, session-1] = np.array([fit['alpha'][1].mean(),  fit['weightAct'][1].mean(), fit['sensitivity'][1].mean()])
            # Mode of estimated paraemter  for Action and color Conditions
            subs_para_mode[run-1,i, 0, session-1] = np.array([find_mode_point(fit['alpha'][0]), find_mode_point(fit['weightAct'][0]), find_mode_point(fit['sensitivity'][0])])
            subs_para_mode[run-1,i, 1, session-1] = np.array([find_mode_point(fit['alpha'][1]), find_mode_point(fit['weightAct'][1]), find_mode_point(fit['sensitivity'][1])])


# main directory
parent_dir = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/simulation/' 
# the grand truth parameters
subs_grand_truth = np.zeros([len(subList), 2, 2, 3])
# loop over list of participants
for i, subName in enumerate(subList):
    for session in [1,2]:
        for cond, condition in enumerate(['Act', 'Stim']): 
            dirc = parent_dir + str(simNumber) + '/' + subName +'/' + subName + '-task-design-true-param.csv'
            simulated_data = pd.read_csv(dirc)
            simulated_data_chunck = simulated_data[(simulated_data['session'] == session) & (simulated_data['block'] == condition)]
            subs_grand_truth[i, cond, session-1] = np.array([simulated_data_chunck['alpha'].unique()[0],
                                                             simulated_data_chunck['weightAct'].unique()[0],
                                                             simulated_data_chunck['beta'].unique()[0]])


# Averaging over mean and mode across runs
subs_para_mean_mean =  subs_para_mean.mean(axis=0)
subs_para_mode_mean =  subs_para_mode.mean(axis=0)


fig = plt.figure(figsize=(15, 10), tight_layout=True)
rows = 2
columns =3 

# sensitivity, (subjs, conds, session, para)
fig.add_subplot(rows, columns, 1)
plt.scatter(subs_para_mode_mean[:,:,:, 2].flatten(),subs_grand_truth[:,:,:, 2].flatten())
plt.plot([.0,.15], [.0,.2])
plt.xlabel('Mode estimation')
plt.ylabel('Grand truth')
plt.title('Sensitivity')
plt.xlim(0, .15)

# learning rate
fig.add_subplot(rows, columns, 2)
plt.scatter(subs_para_mode_mean[:,0,:, 0].flatten(),subs_grand_truth[:,0,:, 0].flatten())
plt.plot([.0,.7], [.0,.7])
plt.xlabel('Mode estimation')
plt.ylabel('Grand truth')
plt.title('Learnig rate for action condition')

# learning rate
fig.add_subplot(rows, columns, 3)
plt.scatter(subs_para_mode_mean[:,1,:, 0].flatten(),subs_grand_truth[:,1,:, 0].flatten())
plt.plot([.0,.7], [.0,.7])
plt.xlabel('Mode estimation')
plt.ylabel('Grand truth')
plt.title('Learnig rate for Color condition')

# weighting
fig.add_subplot(rows, columns, 4)
plt.scatter(subs_para_mode_mean[:,0,:, 1].flatten(),subs_grand_truth[:,0,:, 1].flatten())
plt.plot([.5,1], [.5,1])
plt.xlabel('Mode estimation')
plt.ylabel('Grand truth')
plt.title('Weighting parameter for action condition')

# weighting
fig.add_subplot(rows, columns, 5)
plt.scatter(subs_para_mode_mean[:,1,:, 1].flatten(),subs_grand_truth[:,1,:, 1].flatten())
plt.plot([0,.5], [0,.5])
plt.xlabel('Mode estimation', fontsize=12)
plt.ylabel('Grand truth', fontsize=12)
plt.title('Weighting parameter for color condition')

# save figure
fig.savefig('./figures/para_reco_mode_RL_condition_same_lr.png', dpi=300)


fig = plt.figure(figsize=(15, 10), tight_layout=True)
rows = 2
columns =3 

# sensitivity, (subjs, conds, session, para)
fig.add_subplot(rows, columns, 1)
plt.scatter(subs_para_mean_mean[:,:,:, 2].flatten(),subs_grand_truth[:,:,:, 2].flatten())
plt.plot([.0,.15], [.0,.2])
plt.xlabel('Mean estimation')
plt.ylabel('Grand truth')
plt.title('Sensitivity')
plt.xlim(0, .15)

# learning rate
fig.add_subplot(rows, columns, 2)
plt.scatter(subs_para_mean_mean[:,0,:, 0].flatten(),subs_grand_truth[:,0,:, 0].flatten())
plt.plot([.0,.7], [.0,.7])
plt.xlabel('Mean estimation')
plt.ylabel('Grand truth')
plt.title('Learnig rate for action condition')

# learning rate
fig.add_subplot(rows, columns, 3)
plt.scatter(subs_para_mean_mean[:,1,:, 0].flatten(),subs_grand_truth[:,1,:, 0].flatten())
plt.plot([.0,.7], [.0,.7])
plt.xlabel('Mean estimation')
plt.ylabel('Grand truth')
plt.title('Learnig rate for Color condition')

# weighting
fig.add_subplot(rows, columns, 4)
plt.scatter(subs_para_mean_mean[:,0,:, 1].flatten(),subs_grand_truth[:,0,:, 1].flatten())
plt.plot([.5,1], [.5,1])
plt.xlabel('Mean estimation')
plt.ylabel('Grand truth')
plt.title('Weighting for action condition')

# weighting
fig.add_subplot(rows, columns, 5)
plt.scatter(subs_para_mean_mean[:,1,:, 1].flatten(),subs_grand_truth[:,1,:, 1].flatten())
plt.plot([0,.5], [0,.5])
plt.xlabel('Mean estimation')
plt.ylabel('Grand truth')
plt.title('Weighting for color condition')

# save figure
fig.savefig('./figures/para_reco_mean_RL_condition_same_lr.png', dpi=300)