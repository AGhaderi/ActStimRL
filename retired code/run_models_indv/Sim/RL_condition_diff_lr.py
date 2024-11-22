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
import nest_asyncio

# List of subjects
subList = ['sub-004', 'sub-010', 'sub-012', 'sub-025', 'sub-026', 'sub-029', 'sub-030',
           'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042', 'sub-044', 
           'sub-045', 'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-059', 
           'sub-060', 'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071', 
           'sub-074', 'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 
           'sub-081', 'sub-082', 'sub-083', 'sub-085', 'sub-087', 'sub-088', 'sub-089', 
           'sub-090', 'sub-092', 'sub-108', 'sub-109']
# If you want to model fit or just recall ex model fit
modelFit = True
# Number of chains in MCMC procedure
n_chains = 6
# The number of iteration or samples for each chain in MCM procedure
n_samples=5000
# number of simulation
simNumber = 11
# Main directory of the subject
subMainDirec = '/mnt/scratch/projects/7TPD/amin/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/'
# Main directory of the simupated participatns
parent_dir = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/simulation/'
# Directory of the especifit simulated participant
dirc = parent_dir + str(simNumber) + '/All-simulated-task-design-true-param.csv'
# Read the simulated participant
simulated_data = pd.read_csv(dirc)

# set of indicator to the first trial of each participant
for sub in subList:
    for session in [1, 2]: # session
        for reverse in [21, 14]: # two distinct environemnt
            for condition in ['Act', 'Stim']: # condition
                behAll_indicator = simulated_data[(simulated_data['sub_ID']==sub)&(simulated_data['block']==condition)&(simulated_data['session']==session)&(simulated_data['reverse']==reverse)]  
                simulated_data.loc[(simulated_data['sub_ID']==sub)&(simulated_data['block']==condition)&(simulated_data['session']==session)&(simulated_data['reverse']==reverse), 'indicator'] = np.arange(1, behAll_indicator.shape[0] + 1)
 
# number of conditions
nConds = 2
simulated_data.block = simulated_data.block.replace(['Act', 'Stim'], [1, 2])
# specific subject
for subName in subList:
    # read sessions data for a specifit participant
    for session in [1, 2]: # session
        behAll_sub = simulated_data[(simulated_data['sub_ID']==subName)&(simulated_data['session']==session)]  
        if modelFit == True: 
            """Fitting data to model and then save as pickle file in the subject directory if modelFit = True"""
            # Put required data for stan model
            dataStan = {'N':int(behAll_sub.shape[0]),  
                        'pushed':np.array(behAll_sub.pushed).astype(int),  # should be integer
                        'yellowChosen':np.array(behAll_sub.yellowChosen).astype(int), # should be integer
                        'winAmtPushable':np.array(behAll_sub.winAmtPushable), 
                        'winAmtPullable':np.array(behAll_sub.winAmtPullable),
                        'winAmtYellow':np.array(behAll_sub.winAmtYellow), 
                        'winAmtBlue':np.array(behAll_sub.winAmtPullable),
                        'rewarded':np.array(behAll_sub.correctChoice).astype(int), # should be integer      
                        'indicator':np.array(behAll_sub.indicator).astype(int),  
                        'nConds':nConds,
                        'condition':np.array(behAll_sub.block).astype(int),
                        'p_push_init':.5, 
                        'p_yell_init':.5}
            # initial sampling

            initials = [] 
            for c in range(0, n_chains):
                chaininit = {
                    'alphaAct': np.random.uniform(.1, .5, size=2),
                    'alphClr': np.random.uniform(.1, .5, size=2),        
                    'weightAct': np.random.uniform(.1, .5, size=2),
                    'sensitivity': np.random.uniform(.1, .5, size=2)
                }
                initials.append(chaininit)   

            # Loading the RL Stan Model
            file_name = '/home/amingk/Documents/7TPD/ActStimRL/Model/stan_models/indv/RL_condition.stan' 
            file_read = open(file_name, 'r')
            stan_model = file_read.read()
            # Use nest-asyncio.This package is needed because Jupter Notebook blocks the use of certain asyncio functions
            nest_asyncio.apply()
            # Building Stan Model realted to our proposed model
            posterior = stan.build(stan_model, data = dataStan)
            # Start for taking samples from parameters in the Stan Model
            fit = posterior.sample(num_chains=n_chains, num_samples=n_samples, init=initials)

            # Save Model Fit
            pickelDir = parent_dir + str(simNumber) + '/' + subName + '/sess-' + str(session) + '_RL_condition.pkl'
            utils.to_pickle(stan_fit=fit, save_path = pickelDir)
        else:
            """Loading the pickle file of model fit from the subject directory if modelFit = False"""
            loadPkl = utils.load_pickle(load_path=pickelDir)
            fit = loadPkl['fit']

        # Extracting posterior distributions for each of four main unkhown parameters
        alphaAct_ = fit["alphaAct"]
        alphaClr_ = fit["alphaClr"]
        weightAct_ = fit["weightAct"]
        beta_ = fit["sensitivity"]
 
        # Figure of model fit results in two column and two rows
        fig = plt.figure(figsize=(10, 6), tight_layout=True)
        rows = 2
        columns = 2

        # Weghtening
        fig.add_subplot(rows, columns, 1)
        sns.histplot(weightAct_[0], kde=True, stat='density')
        sns.histplot(weightAct_[1], kde=True, stat='density')
        plt.title(subName + ', Weightening', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.xlabel('$w_{(A)}$', fontsize=14)
        plt.xlim(0, 1)
        plt.legend(['Act', 'Clr']) 

        # Sensitivity
        fig.add_subplot(rows, columns, 2)
        sns.histplot(beta_[0], kde=True, stat='density')
        sns.histplot(beta_[1], kde=True, stat='density')
        plt.title(subName + ', Sensitivity', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.xlabel(r'$\beta$', fontsize=14)
        plt.legend(['Act', 'Clr']) 


        # Action Learning Rate
        fig.add_subplot(rows, columns, 3)
        sns.histplot(alphaAct_[0], kde=True, stat='density')
        sns.histplot(alphaAct_[1], kde=True, stat='density')
        plt.title(subName + ', Action Learning Rate', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.xlabel(r'$ \alpha_{(A)} $', fontsize=14)
        plt.xlim(0, 1)
        plt.legend(['Act', 'Clr']) 

        # Color Learning Rate
        fig.add_subplot(rows, columns, 4)
        sns.histplot(alphaClr_[0], kde=True, stat='density')
        sns.histplot(alphaClr_[1], kde=True, stat='density')
        plt.title(subName + ', Color Learning Rate', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.xlabel(r'$ \alpha_{(C)} $', fontsize=14)
        plt.xlim(0, 1)
        plt.legend(['Act', 'Clr']) 

        plt.subplots_adjust(wspace=10.)

        # Save figure of parameter distribution 
        fig.savefig(parent_dir + str(simNumber) + '/' + subName + '/sess-' + str(session) + '_RL_condition.png', dpi=300)
