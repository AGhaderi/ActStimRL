#!/mrhome/amingk/anaconda3/bin/python
"""Model fit for competing Action Value Learning and Stimulus Value Learning in the cotext of Reinforcement Learning at the individual level"""

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
simNumber = 10
# name of stand plus simuluation number
fileName = 'RL1_sim'
stanName = 'RL1'

# List of subjects
subList = ['sub-004', 'sub-010', 'sub-012', 'sub-025', 'sub-026', 'sub-029', 'sub-030',
           'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042', 'sub-045',
           'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-059', 'sub-060',
           'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071', 'sub-074',
           'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 'sub-081',
           'sub-082', 'sub-083', 'sub-085', 'sub-087', 'sub-088', 'sub-089', 'sub-090',
           'sub-092', 'sub-108', 'sub-109']

# Loop over all subjects
for sub in subList:
    # Number of chains in MCMC procedure
    n_chains = 3
    # The number of iteration or samples for each chain in MCM procedure
    n_samples=2000
    # Name of subject
    subName = sub
    # Number of sessions and conditions
    nCond = 2 
    nSes = 2 

    # Main directory of simulated data
    subMainDirec = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/simulation/' 
    # Directory of the simulated subject
    dirc = subMainDirec + subName + '/' + str(simNumber) + '/' + subName + '-simulated-data-with-task-design-true-param.csv'
    # read simulated dat 
    data = pd.read_csv(dirc)

    # The adrees name of pickle file
    pickelDir = subMainDirec + subName + '/' + str(simNumber) + '/' + subName +'_' + fileName + '.pkl'
    if modelFit == True: 
        """Fitting data to model and then save as pickle file in the subject directory if modelFit = True"""
        # Put required data for stan model
        dataStan = {'N':int(data.shape[0]),  
                    'nCond':nCond, 
                    'nSes':nSes, 
                    'pushed':np.array(data.pushed).astype(int),  
                    'yellowChosen':np.array(data.yellowChosen).astype(int), 
                    'winAmtPushable':np.array(data.winAmtPushable).astype(int), 
                    'winAmtYellow':np.array(data.winAmtYellow).astype(int), 
                    'rewarded':np.array(data.correctChoice).astype(int),       
                    'session':np.array(data.session).astype(int), 
                    'condition':np.array(data.block.replace('Act',1).replace('Stim',2)).astype(int),  
                    'p_push_init':.5, 
                    'p_yell_init':.5}


         # initial sampling
        initials = [] 
        for c in range(0, n_chains):
            chaininit = {
                'alphaAct': np.random.uniform(.4, .6, size=(nSes, nCond)),       
                'alphaClr': np.random.uniform(.4, .6, size=(nSes, nCond)),
                'weightAct': np.random.uniform(.4, .6, size=(nSes, nCond)),   
                'sensitivity': np.random.uniform(0.01, .1, size=(nSes))
            }
            initials.append(chaininit)          

        # Loading the RL Stan Model
        file_name = '../stan_models/' + stanName + '.stan' 
        file_read = open(file_name, 'r')
        stan_model = file_read.read()
        # Use nest-asyncio.This package is needed because Jupter Notebook blocks the use of certain asyncio functions
        nest_asyncio.apply()
        # Building Stan Model realted to our proposed model
        posterior = stan.build(stan_model, data = dataStan)
        # Start for taking samples from parameters in the Stan Model
        fit = posterior.sample(num_chains=n_chains, num_samples=n_samples, init = initials)

        # Save Model Fit
        utils.to_pickle(stan_fit=fit, save_path = pickelDir)
    else:
        """Loading the pickle file of model fit from the subject directory if modelFit = False"""
        loadPkl = utils.load_pickle(load_path=pickelDir)
        fit = loadPkl['fit']

    # Extracting posterior distributions for each of four main unkhown parameters
    weightAct = fit["weightAct"] 
    sensitivity = fit["sensitivity"] 
    alphaAct = fit["alphaAct"] 
    alphaClr = fit["alphaClr"] 


    # Weightening Figure of model fit results in two column and two rows
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    rows = 2
    columns = 2

    fig.add_subplot(rows, columns, 1)
    ax = plt.gca()
    # Get ground truth
    trueValue = np.unique(data.loc[(data['session']==1) & (data['block']=='Act')]['weightAct'])
    plots.plot_posterior(x=weightAct[0,0], 
                         ax=ax, 
                         trueValue=trueValue, 
                         title = subName + ', Ses 1 Act, Weightening',
                         ylabel = 'Density',
                         xlabel = '$w_{(A)}$')
    fig.add_subplot(rows, columns, 2)
    ax = plt.gca()
    trueValue = np.unique(data.loc[(data['session']==1) & (data['block']=='Stim')]['weightAct'])
    plots.plot_posterior(x=weightAct[0,1], 
                         ax=ax, 
                         trueValue=trueValue, 
                         title = subName + ',Ses 1 Clr, Weightening',
                         ylabel = 'Density',
                         xlabel = '$w_{(A)}$')
    fig.add_subplot(rows, columns, 3)
    ax = plt.gca()
    trueValue = np.unique(data.loc[(data['session']==2) & (data['block']=='Act')]['weightAct'])
    plots.plot_posterior(x=weightAct[1,0], 
                         ax=ax, 
                         trueValue=trueValue, 
                         title = subName + ',Ses 2 Act, Weightening',
                         ylabel = 'Density',
                         xlabel = '$w_{(A)}$')
    fig.add_subplot(rows, columns, 4)
    ax = plt.gca()
    trueValue = np.unique(data.loc[(data['session']==2) & (data['block']=='Stim')]['weightAct'])
    plots.plot_posterior(x=weightAct[1,1], 
                         ax=ax, 
                         trueValue=trueValue, 
                         title = subName + ',Ses 2 Clr, Weightening',
                         ylabel = 'Density',
                         xlabel = '$w_{(A)}$')

    fig.savefig(subMainDirec + subName + '/' + str(simNumber) + '/' + subName  +'_' + fileName + '_weightening_sim.png', dpi=300)


    # sensitivity Figure of model fit results in two column and two rows
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    rows = 1
    columns = 2

    fig.add_subplot(rows, columns, 1)
    ax = plt.gca()
    # Get ground truth
    trueValue = np.unique(data.loc[data['session']==1]['beta'])
    plots.plot_posterior(x=sensitivity[0], 
                         ax=ax, 
                         trueValue=trueValue, 
                         title = subName + ', Ses 1, Sensitivity',
                         ylabel = 'Density',
                         xlabel = r'$\beta$')
    fig.add_subplot(rows, columns, 2)
    ax = plt.gca()
    trueValue = np.unique(data.loc[data['session']==2]['beta'])
    plots.plot_posterior(x=sensitivity[1], 
                         ax=ax, 
                         trueValue=trueValue, 
                         title = subName + ',Ses 2, Sensitivity',
                         ylabel = 'Density',
                         xlabel = r'$\beta$')

    fig.savefig(subMainDirec + subName + '/' + str(simNumber) + '/' + subName +'_' + fileName + '_beta_sim.png', dpi=300)


    # Action Learning Rate Figure of model fit results in two column and two rows
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    rows = 2
    columns = 2

    fig.add_subplot(rows, columns, 1)
    ax = plt.gca()
    # Get ground truth
    trueValue = np.unique(data.loc[(data['session']==1) & (data['block']=='Act')]['alphaAct'])
    plots.plot_posterior(x=alphaAct[0,0], 
                         ax=ax, 
                         trueValue=trueValue, 
                         title = subName + ', Ses 1 Act, Action Learning Rate',
                         ylabel = 'Density',
                         xlabel = r'r$\alpha_{(A)}$')
    fig.add_subplot(rows, columns, 2)
    ax = plt.gca()
    trueValue = np.unique(data.loc[(data['session']==1) & (data['block']=='Stim')]['alphaAct'])
    plots.plot_posterior(x=alphaAct[0,1], 
                         ax=ax, 
                         trueValue=trueValue, 
                         title = subName + ',Ses 1 Clr, Action Learning Rate',
                         ylabel = 'Density',
                         xlabel = r'$\alpha_{(A)}$')
    fig.add_subplot(rows, columns, 3)
    ax = plt.gca()
    trueValue = np.unique(data.loc[(data['session']==2) & (data['block']=='Act')]['alphaAct'])
    plots.plot_posterior(x=alphaAct[1,0], 
                         ax=ax, 
                         trueValue=trueValue, 
                         title = subName + ',Ses 2 Act, Action Learning Rate',
                         ylabel = 'Density',
                         xlabel = r'$\alpha_{(A)}$')
    fig.add_subplot(rows, columns, 4)
    ax = plt.gca()
    trueValue = np.unique(data.loc[(data['session']==2) & (data['block']=='Stim')]['alphaAct'])
    plots.plot_posterior(x=alphaAct[1,1], 
                         ax=ax, 
                         trueValue=trueValue, 
                         title = subName + ',Ses 2 Clr, Action Learning Rate',
                         ylabel = 'Density',
                         xlabel = r'$\alpha_{(A)}$')

    fig.savefig(subMainDirec + subName + '/' + str(simNumber) + '/' + subName  +'_' + fileName + '_sim.png', dpi=300)



    # Color Learning Rate Figure of model fit results in two column and two rows
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    rows = 2
    columns = 2

    fig.add_subplot(rows, columns, 1)
    ax = plt.gca()
    # Get ground truth
    trueValue = np.unique(data.loc[(data['session']==1) & (data['block']=='Act')]['alphaClr'])
    plots.plot_posterior(x=alphaClr[0,0], 
                         ax=ax, 
                         trueValue=trueValue, 
                         title = subName + ', Ses 1 Act, Color Learning Rate',
                         ylabel = 'Density',
                         xlabel = r'$\alpha_{(C)}$')
    fig.add_subplot(rows, columns, 2)
    ax = plt.gca()
    trueValue = np.unique(data.loc[(data['session']==1) & (data['block']=='Stim')]['alphaClr'])
    plots.plot_posterior(x=alphaClr[0,1], 
                         ax=ax, 
                         trueValue=trueValue, 
                         title = subName + ',Ses 1 Clr, Color Learning Rate',
                         ylabel = 'Density',
                         xlabel = r'$\alpha_{(C)}$')
    fig.add_subplot(rows, columns, 3)
    ax = plt.gca()
    trueValue = np.unique(data.loc[(data['session']==2) & (data['block']=='Act')]['alphaClr'])
    plots.plot_posterior(x=alphaClr[1,0], 
                         ax=ax, 
                         trueValue=trueValue, 
                         title = subName + ',Ses 2 Act, Color Learning Rate',
                         ylabel = 'Density',
                         xlabel = r'$\alpha_{(C)}$')
    fig.add_subplot(rows, columns, 4)
    ax = plt.gca()
    trueValue = np.unique(data.loc[(data['session']==2) & (data['block']=='Stim')]['alphaClr'])
    plots.plot_posterior(x=alphaClr[1,1], 
                         ax=ax, 
                         trueValue=trueValue, 
                         title = subName + ',Ses 2 Clr, Color Learning Rate',
                         ylabel = 'Density',
                         xlabel = r'$\alpha_{(C)}$')

    fig.savefig(subMainDirec + subName + '/' + str(simNumber) + '/' + subName  + '_' + fileName + '_alphaClr_sim.png', dpi=300)
