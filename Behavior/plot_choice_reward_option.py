import numpy as np
import pandas as pd
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
from Madule import plots

"""Plots of choice responses for each individuals.
In this plot all feasturs of pushing/pulling and yellow chosen/ blue chosen are shown in one plot.
The both choices and rewarded option Are seperated for each feature.
At first you should run 'pool-data.py' to create '_achieva7t_task-DA_beh.csv' for each participant and then use
the current madule to plot choice and rewarded options with together for each participant.
"""

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
    subMainDirec = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/'
    # Directory of main dataset for each participant
    dirc = subMainDirec + subName + '/' + subName + '_achieva7t_task-DA_beh.csv'
    # Read the excel file
    data = pd.read_csv(dirc)
    # Condition sequences for each particiapnt
    blocks = data.groupby(['session', 'run'])['block'].unique().to_numpy()
    blocks = np.array([blocks[0], blocks[1], blocks[2], blocks[3]]).flatten()
    #save file name
    saveFile = subMainDirec + subName + '/' + subName + '_achieva7t_task-DA_beh.png'
    # Plot by a pre implemented madule
    plots.plotChosenCorrect(data = data, blocks = blocks, subName = subName, saveFile = saveFile)
    #plotChosenCorrect_modofied1, plotChosenCorrect_modofied2, plotChosenCorrect_modofied3, plotChosenCorrect_modofied4, plotChosenCorrect_modofied5