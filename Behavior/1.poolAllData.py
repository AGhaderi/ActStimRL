import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
import glob

"""For each condition there is a reversal point indicating chaning environment and probability of reward.
Reversal points denoted by 14 representing change of environemnt after each 14 trials, so there are two reversal points for 42 trials , 
and 21 representing change of the environemnt for each 21 trials, so there is one  reversal point for 42 trials. Then I collected all sesseions and runs 
files into on csv file.
group label 1: PD OFF
group label 2: HC
group label 3: PD ON
"""
# List of subjects
subList = ['sub-004','sub-005', 'sub-010', 'sub-012', 'sub-025', 'sub-026', 'sub-029', 'sub-030',
           'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042', 'sub-044', 'sub-045',
           'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-057', 'sub-059', 'sub-060',
           'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071', 'sub-074', 'sub-075', 
           'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 'sub-081', 'sub-082', 'sub-083', 
           'sub-085',  'sub-086', 'sub-087', 'sub-088', 'sub-089', 'sub-090', 'sub-091', 'sub-092', 
           'sub-106',  'sub-108', 'sub-109', 'sub-121']


# Group labels including 1, 2, 3
randomGroupLabel = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/randomGroupLabel.csv')
# Main director
mainDirec = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/'

# Concatenate all participatns excel files into a one dataframe
dataAll = pd.DataFrame([])
# List of subjects
for subName in subList:
    # Read out group label for each session
    labelSes1 = int(randomGroupLabel.loc[randomGroupLabel['sub-ID'] == subName, 'ses-02'])
    labelSes2 = int(randomGroupLabel.loc[randomGroupLabel['sub-ID'] == subName, 'ses-03'])
    
    for sess in ['ses-02achieva7t', 'ses-03achieva7t']:
        # List of existing .csv files for each session and run realted to the subject
        files = glob.glob(f'{mainDirec}/{subName}/{sess}/beh/*.txt')

        if len(files)!=0:
            # Concatenated all four .csv file for a specific subject
            data = pd.DataFrame([])
            for i in range(len(files)):
                df = pd.read_csv(files[i], sep=r'[\t,]', engine='python')
                data = pd.concat([data, df])

            # Set participants label into the dataframe
            data['sub_ID'] = subName
            # Set group label into the dataframe
            data.loc[data['session'] == 1, 'group'] = str(labelSes1)
            data.loc[data['session'] == 2, 'group'] = str(labelSes2)
            # Set patient label, helathy control or parkindon's disease
            if labelSes1==2:
                data['patient'] = 'HC'
            if labelSes1!=2:
                data['patient'] = 'PD'

            #group label 1: PD OFF, group label 3: PD ON, group 2: HC (OFF again)
            data['medication'] = data['group'].replace(['1','2','3'], ['OFF', 'OFF', 'ON'])

            ################################### Set inverse columns
            files = glob.glob(f'{mainDirec}/{subName}/{sess}/beh/*_beh.mat')
            data_mat = loadmat(files[0])
            blockList1_1 = data_mat['blockList1_1'][0][0]
            blockList1_2 = data_mat['blockList1_2'][0][0]
            blockList2_1 = data_mat['blockList2_1'][0][0]
            blockList2_2 = data_mat['blockList2_2'][0][0]
            # Concatenate two sessions
            data_reverse = np.array([blockList1_1, blockList1_2, blockList2_1, blockList2_2])
            
            # Condition sequences for each particiapnt
            blocks = data.groupby(['run'])['block'].unique().to_numpy()
            # number of runs
            runs = data['run'].unique()
            if len(runs)==2: # both run 1 and 2 are available
                blocks = np.array([blocks[0][0], blocks[0][1], blocks[1][0], blocks[1][1]])
            else: # run 1 or run2 is available
                blocks = np.array([blocks[0][0], blocks[0][1]])
            # 21 reversal point has two phases called stable environment, 14 reversal point has three phases called volatile environment
            idx = 0
            data['reverse'] = np.nan
            data['phase'] = np.nan
            for run in runs: # two run
                for b in range(1, 3): # two condition
                    # add reversal point matched condition, each condition in on run and session has one environemnet
                    data.loc[(data['run']==run)&(data['block']==blocks[idx]), 'reverse'] = data_reverse[idx]
                    # Adding phases
                    # There is a exception for a participant which the number of trial for a run is not 42 but 27
                    if ((data['run']==run)&(data['block']==blocks[idx])).sum()==27:
                        phase = np.repeat(['phase1', 'phase2'], 21)[0:27]
                        data.loc[(data['run']==run)&(data['block']==blocks[idx]), 'phase'] = phase
                    elif data_reverse[idx] == 21:
                        phase = np.repeat(['phase1', 'phase2'], 21)
                        data.loc[(data['run']==run)&(data['block']==blocks[idx]), 'phase'] = phase
                    elif data_reverse[idx] == 14:
                        phase = np.repeat(['phase1', 'phase2', 'phase3'], 14)
                        data.loc[(data['run']==run)&(data['block']==blocks[idx]), 'phase'] = phase
                    idx +=1

            # Concatenating dataframe of particiatns into one general dataframe for further uses
            dataAll = pd.concat([dataAll, data])

# Save all raw data from participants 
dataAll.to_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/rawBehAll.csv', index=False)
 
# Detection of irregular responces (no-responses or error responces)
temp = dataAll['pushed'].to_numpy().astype(int)
NoNanBehAll = dataAll[temp>=0]
# set of indicator to the first trial of each blok in each run. For Future RL Modeling Purpose
for sub in subList:
    for session in [1, 2]: # session 1 and 2
        for run in [1, 2]: # two runs in each session
            for condition in ['Act', 'Stim']: # two conditions
                behAll_indicator = NoNanBehAll[(NoNanBehAll['sub_ID']==sub)&(NoNanBehAll['block']==condition)&(NoNanBehAll['session']==session)&(NoNanBehAll['run']==run)]  
                NoNanBehAll.loc[(NoNanBehAll['sub_ID']==sub)&(NoNanBehAll['block']==condition)&(NoNanBehAll['session']==session)&(NoNanBehAll['run']==run), 'indicator'] = np.arange(1, behAll_indicator.shape[0] + 1)


# Run 2 Action condition of session2 of sub-010 is not complete
NoNanBehAll = NoNanBehAll[(NoNanBehAll['sub_ID']!='sub-010')|(NoNanBehAll['session']!=2)|(NoNanBehAll['run']!=2)|(NoNanBehAll['block']!='Act')]
# Run 2 session 2 of sub-030 has lots of no response and should be exluded.
NoNanBehAll = NoNanBehAll[(NoNanBehAll['sub_ID']!='sub-030')|(NoNanBehAll['session']!=2)|(NoNanBehAll['run']!=2)]
# Run 2 session 1  of sub-086 has lots of no responses and should be exluced
NoNanBehAll = NoNanBehAll[(NoNanBehAll['sub_ID']!='sub-086')|(NoNanBehAll['session']!=1)|(NoNanBehAll['run']!=2)]

# first exclusion based on the data corruption
withdraw_subs1 = ['sub-057', 'sub-076', 'sub-083', 'sub-091', 'sub-106']
for sub in withdraw_subs1:
    NoNanBehAll = NoNanBehAll[NoNanBehAll['sub_ID']!=sub] 
# Save all no nan and outlier data from participants 
NoNanBehAll.to_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/NoNanBehAll.csv', index=False)



