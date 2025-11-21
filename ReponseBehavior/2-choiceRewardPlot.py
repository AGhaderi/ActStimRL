import numpy as np
import pandas as pd
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
from Madule import plots
import glob
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt 


"""Plots of choice responses for each individuals.
In this plot all feasturs of pushing/pulling and yellow chosen/ blue chosen are shown in one plot.
The both choices and rewarded option Are seperated for each feature.
At first you should run 'pool-data.py' to create '_achieva7t_task-DA_beh.csv' for each participant and then use
the current madule to plot choice and rewarded options with together for each participant.
"""

# List of subjects
subList = ['sub-004','sub-005', 'sub-010', 'sub-012', 'sub-025', 'sub-026', 'sub-029', 'sub-030',
           'sub-033', 'sub-034', 'sub-036', 'sub-040', 'sub-041', 'sub-042', 'sub-044', 'sub-045',
           'sub-047', 'sub-048', 'sub-052', 'sub-054', 'sub-056', 'sub-057', 'sub-059', 'sub-060',
           'sub-064', 'sub-065', 'sub-067', 'sub-069', 'sub-070', 'sub-071', 'sub-074', 'sub-075', 
           'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 'sub-081', 'sub-082', 'sub-083', 
           'sub-085',  'sub-086', 'sub-087', 'sub-088', 'sub-089', 'sub-090', 'sub-091', 'sub-092', 
           'sub-106',  'sub-108', 'sub-109', 'sub-121']

 
# main directory
mainDirec = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA'
for subName in subList:
    for sess in ['ses-02achieva7t', 'ses-03achieva7t']:
            # List of existing .csv files for each session and run realted to the subject
            files = glob.glob(f'{mainDirec}/{subName}/{sess}/beh/*.txt')

            if len(files)!=0:
                # Concatenated all four .txt file for a specific subject
                data = pd.DataFrame([])
                for i in range(len(files)):
                    df = pd.read_csv(files[i], sep=r'[\t,]', engine='python')
                    
                    # Get the filename of the currently running script
                    filenameExt = os.path.basename(files[i])
                    # Remove the .py extension from the filename
                    file_name = os.path.splitext(filenameExt)[0]
                    # concatenate
                    data = pd.concat([data, df])

                # List of existing .mat files for each session and run realted to the subject
                files = glob.glob(f'{mainDirec}/{subName}/{sess}/beh/*_beh.mat')
                data_mat = loadmat(files[0])
                blockList1_1 = data_mat['blockList1_1'][0][0]
                blockList1_2 = data_mat['blockList1_2'][0][0]
                blockList2_1 = data_mat['blockList2_1'][0][0]
                blockList2_2 = data_mat['blockList2_2'][0][0]
                # Concatenate two sessions
                data_reverse = np.array([blockList1_1, blockList1_2, blockList2_1, blockList2_2])

                # Get the filename of the currently running script
                filenameExt = os.path.basename(files[0])
                # Remove the .py extension from the filename
                filename = os.path.splitext(filenameExt)[0]
                #save file name
                saveFile = f'{mainDirec}/{subName}/{sess}/beh/analysis/{subName}_{sess}_task-DA_beh.png'
                # Check out if it does not exist
                if not os.path.isdir(f'{mainDirec}/{subName}/{sess}/beh/analysis/'):
                        os.makedirs(f'{mainDirec}/{subName}/{sess}/beh/analysis/') 
                # Plot by a pre implemented madule
                plots.plotChosenCorrect(data = data, subName = subName, reverse =  data_reverse, saveFile = saveFile)




 