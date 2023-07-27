#!/mrhome/amingk/anaconda3/bin/python

import numpy as np
import pandas as pd
import stan
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
sys.path.append('..')
from madule import utils
from madule import plots
import nest_asyncio

# If you want to model fit or just recall ex model fit
modelFit = True

# Number of simulation
simNumber = 1

# Number of chains in MCMC procedure
nChains = 10
# The number of iteration or samples for each chain in MCM procedure
nSamples = 3000
# Main directory of simulated data
subMainDirec = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/simulation/' 
# Directory of the simulated subject
dirc = subMainDirec + 'hierParam/'  + str(simNumber) + '/' +'hier-simulated-data-with-task-design-true-param.csv'
# read simulated dat 
data = pd.read_csv(dirc)
    
# The adrees name of pickle file
pickelDir = subMainDirec + 'hierParam/'  + str(simNumber) + '/'  +'hier_RL_hier_3_sim.pkl'
if modelFit == True: 
    """Fitting data to model and then save as pickle file in the subject directory if modelFit = True"""
    # data from group label 2 particiapnt
    dataGroup2 = data.loc[data['group']==2]   

    nCond = 2 
    nSes = 2 
    # number of participant
    nParts = len(np.unique(dataGroup2.sub_ID))
    # participant indeces
    participant = dataGroup2.sub_ID.replace(np.unique(dataGroup2.sub_ID),
                          np.arange(1, nParts +1, 1))
    # condition indeces
    condition = dataGroup2.block.replace('Act',1).replace('Stim',2)

    # Put required data for stan model
    dataStan = {'N':int(dataGroup2.shape[0]),  
                'nParts': nParts,
                'nCond':2, 
                'nSes':2, 
                'pushed':np.array(dataGroup2.pushed).astype(int),  
                'yellowChosen':np.array(dataGroup2.yellowChosen).astype(int), 
                'winAmtPushable':np.array(dataGroup2.winAmtPushable).astype(int), 
                'winAmtYellow':np.array(dataGroup2.winAmtYellow).astype(int), 
                'rewarded':np.array(dataGroup2.correctChoice).astype(int),  
                'p_push_init':.5, 
                'p_yell_init':.5,        
                'participant':np.array(participant).astype(int),      
                'session':np.array(dataGroup2.session).astype(int),
                'condition':np.array(condition).astype(int)}    
    
    
    # initial sampling
    initials = [] 
    for c in range(0, nChains):
        chaininit = {
            'alphaAct_sd': np.random.uniform(.05, .2),
            'alphaClr_sd': np.random.uniform(.05, .2),        
            'weightAct_sd': np.random.uniform(.05, .2),
            'sensitivity_sd': np.random.uniform(.05, .2),
            'alphaAct_hier': np.random.uniform(.3, .7, size=(nSes, nCond)),
            'alphaClr_hier': np.random.uniform(.3, .7, size=(nSes, nCond)),
            'weightAct_hier': np.random.uniform(.3, .7, size=(nSes, nCond)),        
            'sensitivity_hier': np.random.uniform(.1, .1, size=(nSes)),
            'alphaAct': np.random.uniform(.1, .9, size=(nParts, nSes, nCond)),       
            'alphaClr': np.random.uniform(.1, .9, size=(nParts, nSes, nCond)),
            'weightAct': np.random.uniform(.1, .9, size=(nParts, nSes, nCond)),   
            'sensitivity': np.random.uniform(.1, .2, size=(nParts, nSes))
        }
        initials.append(chaininit)   

    # Loading the RL Stan Model
    file_name = '../stan_models/RL_hier_3.stan' 
    file_read = open(file_name, 'r')
    stan_model = file_read.read()
    # Use nest-asyncio.This package is needed because Jupter Notebook blocks the use of certain asyncio functions
    nest_asyncio.apply()
    # Building Stan Model realted to our proposed model
    posterior = stan.build(stan_model, data = dataStan)
    # Start for taking samples from parameters in the Stan Model
    fit = posterior.sample(num_chains=nChains, num_samples=nSamples, init = initials)
    # Save Model Fit
    utils.to_pickle(stan_fit=fit, save_path = pickelDir)
else:
    """Loading the pickle file of model fit from the subject directory if modelFit = False"""
    loadPkl = utils.load_pickle(load_path=pickelDir)
    fit = loadPkl['fit']
    
# Extracting posterior distributions for each of four main unkhown parameters
weightAct = fit["weightAct_hier"] 
sensitivity = fit["sensitivity_hier"] 
alphaAct = fit["alphaAct_hier"] 
alphaClr = fit["alphaClr_hier"] 

# Read hierarchical true parameters
hierFile = subMainDirec + 'hierParam/'  + str(simNumber) + '/'  +'hier-Mean-Std-True-Param.csv'
dataHier = pd.read_csv(hierFile)

# Weightening Figure of model fit results in two column and two rows
fig = plt.figure(figsize=(10, 6), tight_layout=True)
rows = 2
columns = 2

fig.add_subplot(rows, columns, 1)
ax = plt.gca()
# Get ground truth
trueValue = dataHier['hierAlphaAct_mu'][0]
plots.plot_posterior(x=weightAct[0,0], 
                     ax=ax, 
                     trueValue=trueValue, 
                     title = 'Ses 1 Act, Hierarchical Weightening',
                     ylabel = 'Density',
                     xlabel = '$w_{(A)}$')
fig.add_subplot(rows, columns, 2)
ax = plt.gca()
trueValue = dataHier['hierAlphaAct_mu'][1]
plots.plot_posterior(x=weightAct[0,1], 
                     ax=ax, 
                     trueValue=trueValue, 
                     title = 'Ses 1 Clr, Hierarchical Weightening',
                     ylabel = 'Density',
                     xlabel = '$w_{(A)}$')
fig.add_subplot(rows, columns, 3)
ax = plt.gca()
trueValue = dataHier['hierAlphaAct_mu'][2]
plots.plot_posterior(x=weightAct[1,0], 
                     ax=ax, 
                     trueValue=trueValue, 
                     title = 'Ses 2 Act, Hierarchical Weightening',
                     ylabel = 'Density',
                     xlabel = '$w_{(A)}$')
fig.add_subplot(rows, columns, 4)
ax = plt.gca()
trueValue = dataHier['hierAlphaAct_mu'][3]
plots.plot_posterior(x=weightAct[1,1], 
                     ax=ax, 
                     trueValue=trueValue, 
                     title = 'Ses 2 Clr, Hierarchical Weightening',
                     ylabel = 'Density',
                     xlabel = '$w_{(A)}$')

fig.savefig(subMainDirec + 'hierParam/'  + str(simNumber) + '/'  +'hier_RL_hier_3_weightening_sim.png', dpi=300)
 
# sensitivity Figure of model fit results in two column and two rows
fig = plt.figure(figsize=(10, 6), tight_layout=True)
rows = 1
columns = 2

fig.add_subplot(rows, columns, 1)
ax = plt.gca()
# Get ground truth
trueValue = dataHier['hierbeta_mu'][0]
plots.plot_posterior(x=sensitivity[0], 
                     ax=ax, 
                     trueValue=trueValue, 
                     title = 'Ses 1, Hierarchical Weightening',
                     ylabel = 'Density',
                     xlabel = '$w_{(A)}$')
fig.add_subplot(rows, columns, 2)
ax = plt.gca()
trueValue = dataHier['hierbeta_mu'][1]
plots.plot_posterior(x=sensitivity[1], 
                     ax=ax, 
                     trueValue=trueValue, 
                     title = 'Ses 1, Hierarchical Weightening',
                     ylabel = 'Density',
                     xlabel = '$w_{(A)}$')

fig.savefig(subMainDirec + 'hierParam/'  + str(simNumber) + '/'  +'hier_RL_hier_3_sim.png', dpi=300)


# Action Learning Rate Figure of model fit results in two column and two rows
fig = plt.figure(figsize=(10, 6), tight_layout=True)
rows = 2
columns = 2

fig.add_subplot(rows, columns, 1)
ax = plt.gca()
# Get ground truth
trueValue = dataHier['hierAlphaAct_mu'][0]
plots.plot_posterior(x=alphaAct[0,0], 
                     ax=ax, 
                     trueValue=trueValue, 
                     title = 'Ses 1 Act, Hierarchical Action Learning Rate',
                     ylabel = 'Density',
                     xlabel = '$w_{(A)}$')
fig.add_subplot(rows, columns, 2)
ax = plt.gca()
trueValue = dataHier['hierAlphaAct_mu'][1]
plots.plot_posterior(x=alphaAct[0,1], 
                     ax=ax, 
                     trueValue=trueValue, 
                     title = 'Ses 1 Clr, Hierarchical Action Learning Rate',
                     ylabel = 'Density',
                     xlabel = '$w_{(A)}$')
fig.add_subplot(rows, columns, 3)
ax = plt.gca()
trueValue = dataHier['hierAlphaAct_mu'][2]
plots.plot_posterior(x=alphaAct[1,0], 
                     ax=ax, 
                     trueValue=trueValue, 
                     title = 'Ses 2 Act, Hierarchical Action Learning Rate',
                     ylabel = 'Density',
                     xlabel = '$w_{(A)}$')
fig.add_subplot(rows, columns, 4)
ax = plt.gca()
trueValue = dataHier['hierAlphaAct_mu'][3]
plots.plot_posterior(x=alphaAct[1,1], 
                     ax=ax, 
                     trueValue=trueValue, 
                     title = 'Ses 2 Clr, Hierarchical Action Learning Rate',
                     ylabel = 'Density',
                     xlabel = '$w_{(A)}$')

fig.savefig(subMainDirec + 'hierParam/'  + str(simNumber) + '/'  +'hier_RL_hier_3_alphaAct_sim.png', dpi=300)



# Color Learning Rate Figure of model fit results in two column and two rows
fig = plt.figure(figsize=(10, 6), tight_layout=True)
rows = 2
columns = 2

fig.add_subplot(rows, columns, 1)
ax = plt.gca()
# Get ground truth
trueValue = dataHier['hierAlphaClr_mu'][0]
plots.plot_posterior(x=alphaClr[0,0], 
                     ax=ax, 
                     trueValue=trueValue, 
                     title = 'Ses 1 Act, Hierarchical Color Learning Rate',
                     ylabel = 'Density',
                     xlabel = '$w_{(A)}$')
fig.add_subplot(rows, columns, 2)
ax = plt.gca()
trueValue = dataHier['hierAlphaClr_mu'][1]
plots.plot_posterior(x=alphaClr[0,1], 
                     ax=ax, 
                     trueValue=trueValue, 
                     title = 'Ses 1 Clr, Hierarchical Color Learning Rate',
                     ylabel = 'Density',
                     xlabel = '$w_{(A)}$')
fig.add_subplot(rows, columns, 3)
ax = plt.gca()
trueValue = dataHier['hierAlphaClr_mu'][2]
plots.plot_posterior(x=alphaClr[1,0], 
                     ax=ax, 
                     trueValue=trueValue, 
                     title = 'Ses 2 Act, Hierarchical Color Learning Rate',
                     ylabel = 'Density',
                     xlabel = '$w_{(A)}$')
fig.add_subplot(rows, columns, 4)
ax = plt.gca()
trueValue = dataHier['hierAlphaClr_mu'][3]
plots.plot_posterior(x=alphaClr[1,1], 
                     ax=ax, 
                     trueValue=trueValue, 
                     title = 'Ses 2 Clr, Hierarchical Color Learning Rate',
                     ylabel = 'Density',
                     xlabel = '$w_{(A)}$')

fig.savefig(subMainDirec + 'hierParam/'  + str(simNumber) + '/'  +'hier_RL_hier_3_alphaClr_sim.png', dpi=300)
