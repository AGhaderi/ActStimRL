#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python
import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from Madule import simulation_same_lr_switch_amount
from Madule import plots 
 
" Simulation number"
simNumber = 3
"Set mean and STD of Learning rate for Action Value Conditions"
alpha_mu = np.array([[.3, .3], [.3, .3]])
alpha_sd = np.array(.2)

"Set mean and STD of Relative Contribution Parameter"
weightAct_mu = np.array([[.8,.2], [.8,.2]])
weightAct_sd = np.array(.2)

"Set mean and STD of Sensitivity Parameter"
beta_mu = np.array([[.03, .03], [.03, .03]])
beta_sd = np.array(0.05) 

"True values of individual-level parameters are randomly taken from predefined hierarchical level parameters"
simulation_same_lr_switch_amount.set_true_all_parts(alpha_mu_arg=alpha_mu, alpha_sd_arg=alpha_sd,
                                                    weightAct_mu_arg=weightAct_mu, weightAct_sd_arg=weightAct_sd,
                                                    beta_mu_arg=beta_mu, beta_sd_arg=beta_sd,
                                                    simNumber_arg=simNumber)
                      
# simulation run
runNumber = 11
for run in range(1, runNumber):
    # The Final step is to simulate data from the grand truth parameters that has been generated from previous step
    simulation_same_lr_switch_amount.simulate_data_true_params(simNumber=simNumber, runNumber= run)

    """Pooling data all data and then save it"""
    # List of subjects
    subList = ['sub-004', 'sub-010', 'sub-012', 'sub-025', 'sub-026', 'sub-029', 'sub-030',
                'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042', 'sub-044', 
                'sub-045', 'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-059', 
                'sub-060', 'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071', 
                'sub-074', 'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 
                'sub-081', 'sub-082', 'sub-083', 'sub-085', 'sub-087', 'sub-088', 'sub-089', 
                'sub-090', 'sub-092', 'sub-108', 'sub-109']
    # Dataframe for concatenating data
    dataAll = pd.DataFrame([])
    # Loop over list of participatns
    for subName in subList:
        # Main directory of the simupated participatns
        parent_dir = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/simulation/'
        # Directory of the especifit simulated participant
        dirc = parent_dir + str(simNumber) + '/' + subName + '/' +subName +'-simulated-task-design-true-param'+str(run)+'.csv'
        # Read the simulated participant
        data = pd.read_csv(dirc)
        # Set the new column of the participants' name
        data['sub_ID'] = subName
    
        # Concatenating each dataframe
        dataAll = pd.concat([dataAll, data])    
        
    # Save concatenated data over all particiapnts
    dataAll.to_csv(parent_dir + str(simNumber) + '/' +'All-simulated-task-design-true-param'+str(run)+'.csv', index=False)


    # choice plot
    # List of subjects
    subList = ['sub-004', 'sub-010', 'sub-012', 'sub-025', 'sub-026', 'sub-029', 'sub-030',
            'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042', 'sub-044', 
            'sub-045', 'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-059', 
            'sub-060', 'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071', 
            'sub-074', 'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 
            'sub-081', 'sub-082', 'sub-083', 'sub-085', 'sub-087', 'sub-088', 'sub-089', 
            'sub-090', 'sub-092', 'sub-108', 'sub-109']
    
    for subName in subList:
        # Main directory of the subject
        parent_dir = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/simulation/'
        # Directory of the especifit simulated participant
        dirc = parent_dir + str(simNumber) + '/' + subName + '/' +subName +'-simulated-task-design-true-param'+str(run)+'.csv'
        # Read the excel file
        data = pd.read_csv(dirc)
        # Condition sequences for each particiapnt
        blocks = data.groupby(['session', 'run'])['block'].unique().to_numpy()
        blocks = np.array([blocks[0], blocks[1], blocks[2], blocks[3]]).flatten()
        #save file name
        saveFile = parent_dir + str(simNumber) + '/' + subName + '/' +subName +'-achieva7t_task-DA_beh'+str(run)+'.png'

        # Plot by a pre implemented madule
        plots.plotChosenCorrect(data = data, blocks = blocks, subName = subName, saveFile = saveFile)