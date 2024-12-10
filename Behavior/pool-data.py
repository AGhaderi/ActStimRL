import numpy as np
import pandas as pd
from scipy.io import loadmat

"""For each condition there is a reversal point indicating chaning environment and probability of reward.
Reversal points denoted by 14 representing change of environemnt after each 14 trials, so there are two reversal points for 42 trials , 
and 21 representing change of the environemnt for each 21 trials, so there is one  reversal point for 42 trials. Then I collected all sesseions and runs 
files into on csv file.
group label 1: PD OFF
group label 2: HC
group label 3: PD ON
"""
# List of subjects
subList = ['sub-004', 'sub-010', 'sub-012', 'sub-025', 'sub-026', 'sub-029', 'sub-030',
           'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042', 'sub-044', 
           'sub-045', 'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-059', 
           'sub-060', 'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071', 
           'sub-074', 'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 
           'sub-081', 'sub-082', 'sub-083', 'sub-085', 'sub-087', 'sub-088', 'sub-089', 
           'sub-090', 'sub-092', 'sub-108', 'sub-109']

# Group labels including 1, 2, 3
randomGroupLabel = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/randomGroupLabel.csv')
# Concatenate all participatns excel files into a one dataframe
dataAll = pd.DataFrame([])
# List of subjects
for subName in subList:
    # List of existing .csv files for each session and run realted to the subject
    files = ['/ses-02achieva7t/' + subName + '_ses-02achieva7t_task-DA_run-1_beh',
             '/ses-02achieva7t/' + subName + '_ses-02achieva7t_task-DA_run-2_beh',
             '/ses-03achieva7t/' + subName + '_ses-03achieva7t_task-DA_run-1_beh',
             '/ses-03achieva7t/' + subName + '_ses-03achieva7t_task-DA_run-2_beh']

    # Main directory
    subMainDirec = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/'
    # Concatenated all four .csv file for a specific subject
    data = pd.DataFrame([])
    for i in range(len(files)):
        dirc = subMainDirec + subName + files[i] + '.csv'
        df = pd.read_csv(dirc)
        data = pd.concat([data, df])
    
    # Condition sequences for each particiapnt
    blocks = data.groupby(['session', 'run'])['block'].unique().to_numpy()
    blocks = np.array([blocks[0], blocks[1], blocks[2], blocks[3]]).flatten()

    # Reversal point for each participatn
    #  14 incicates three phases in each condition, 21 inicates two phases in each condition
    #  .mat files includes all information, we just use it for phase sequences
    # Session 1 
    data_reverse_session1 = loadmat(subMainDirec + subName + files[0] + '.mat')
    sess1_blockList1_1 = data_reverse_session1['blockList1_1'][0][0]
    sess1_blockList1_2 = data_reverse_session1['blockList1_2'][0][0]
    sess1_blockList2_1 = data_reverse_session1['blockList2_1'][0][0]
    sess1_blockList2_2 = data_reverse_session1['blockList2_2'][0][0]
    # Session 2
    data_reverse_session2 = loadmat(subMainDirec + subName + files[2] + '.mat')
    sess2_blockList1_1 = data_reverse_session2['blockList1_1'][0][0]
    sess2_blockList1_2 = data_reverse_session2['blockList1_2'][0][0]
    sess2_blockList2_1 = data_reverse_session2['blockList2_1'][0][0]
    sess2_blockList2_2 = data_reverse_session2['blockList2_2'][0][0]
    # Concatenate two sessions
    data_reverse = np.array([sess1_blockList1_1, sess1_blockList1_2, sess1_blockList2_1, sess1_blockList2_2,
                             sess2_blockList1_1, sess2_blockList1_2, sess2_blockList2_1, sess2_blockList2_2])
    
    # Add new column to the original data indicating phase and reversal point
    # 21 reversal point has two phases called stable environment, 14 reversal point has three phases called volatile environment
    data['reverse'] = np.nan
    data['phase'] = np.nan
    idx = 0
    for s in range(1, 3): # two session
        for r in range(1, 3): # two run
            for b in range(1, 3): # two condition
                # add reversal point matched condition, each condition in on run and session has one environemnet
                data.loc[(data['session']==s)&(data['run']==r)&(data['block']==blocks[idx]), 'reverse'] = data_reverse[idx]
                # Adding phases
                # There is a exception for a participant which the number of trial for a run is not 42 but 27
                if ((data['session']==s)&(data['run']==r)&(data['block']==blocks[idx])).sum()==27:
                    phase = np.repeat(['phase1', 'phase2'], 21)[0:27]
                    data.loc[(data['session']==s)&(data['run']==r)&(data['block']==blocks[idx]), 'phase'] = phase
                elif data_reverse[idx] == 21:
                    phase = np.repeat(['phase1', 'phase2'], 21)
                    data.loc[(data['session']==s)&(data['run']==r)&(data['block']==blocks[idx]), 'phase'] = phase
                elif data_reverse[idx] == 14:
                    phase = np.repeat(['phase1', 'phase2', 'phase3'], 14)
                    data.loc[(data['session']==s)&(data['run']==r)&(data['block']==blocks[idx]), 'phase'] = phase
                idx +=1
                
    # Set participants label into the dataframe
    data['sub_ID'] = subName
    # Read out group label for each session
    labelSes1 = int(randomGroupLabel.loc[randomGroupLabel['sub-ID'] == subName, 'ses-02'])
    labelSes2 = int(randomGroupLabel.loc[randomGroupLabel['sub-ID'] == subName, 'ses-03'])
    # Set group label into the dataframe
    data.loc[data['session'] == 1, 'group'] = str(labelSes1)
    data.loc[data['session'] == 2, 'group'] = str(labelSes2)
    # Set patient label, helathy control or parkindon's disease
    if labelSes1==2:
        data['patient'] = 'HC'
    if labelSes1!=2:
        data['patient'] = 'PD'

    # Save dataframe for each participant seperately
    saveFile = subMainDirec + subName + '/' + subName + '_achieva7t_task-DA_beh.csv'
    data.to_csv(saveFile ,index=False)    

    # Concatenating dataframe of particiatns into one general dataframe for further uses
    dataAll = pd.concat([dataAll, data])

# Detection of irregular responces (no-responses or error responces)
temp = dataAll['pushed'].to_numpy().astype(int)
dataAllClear = dataAll[temp>=0]
# Save all cleaned data from participants 
dataAllClear.to_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/behAll.csv', index=False)

# Save all raw data from participants 
dataAll.to_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/rawBehAll.csv', index=False)