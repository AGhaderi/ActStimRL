import numpy as np #
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
# read collected data across data
behAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/rawBehAll.csv')


actPropopr_ActCond = np.zeros((len(subList), 2, 5), dtype=float) # number of subjects* 2 sessions* 10 phases for 2 runs and two conditions in each session
clrPropopr_ActCond = np.zeros((len(subList), 2, 5), dtype=float) # number of subjects* 2 sessions* 10 phases for 2 runs and two conditions in each session
clrPropor_ClrtCond = np.zeros((len(subList), 2, 5), dtype=float) # number of subjects* 2 sessions* 10 phases for 2 runs and two conditions in each session
actPropor_ClrCond = np.zeros((len(subList), 2, 5), dtype=float) # number of subjects* 2 sessions* 10 phases for 2 runs and two conditions in each session

# loop over all participants, run,
for i, subName in enumerate(subList):
    for sess in range(2):
        j = 0
        k=0
        for run in range(2):
            for cond in ['Stim', 'Act']:
                if cond=='Act':
                    # take data from specific session, run for a subject
                    actData = behAll[(behAll['session']==sess+1) & (behAll['run']==run+1) & (behAll['block']==cond) & (behAll['sub_ID']==subName)]
                    # if the reversal point is 21
                    if actData['reverse'].unique()==21:
                        # Phase 1
                        actPropopr_ActCond[i, sess, j] = actData[actData['phase']=='phase1']['pushCorrect'].mean()
                        clrPropopr_ActCond[i, sess, j] = actData[actData['phase']=='phase1']['yellowCorrect'].mean()
                        j +=1
                        
                        # Phase 2
                        actPropopr_ActCond[i, sess, j] = actData[actData['phase']=='phase2']['pushCorrect'].mean()
                        clrPropopr_ActCond[i, sess, j] = actData[actData['phase']=='phase2']['yellowCorrect'].mean()
                        j +=1
                    # if the reversal point is 14
                    elif actData['reverse'].unique()==14:
                        
                        # Phase 1
                        actPropopr_ActCond[i, sess, j] = actData[actData['phase']=='phase1']['pushCorrect'].mean()
                        clrPropopr_ActCond[i, sess, j] = actData[actData['phase']=='phase1']['yellowCorrect'].mean()
                        j +=1
                        
                        # Phase 2
                        actPropopr_ActCond[i, sess, j] = actData[actData['phase']=='phase2']['pushCorrect'].mean()
                        clrPropopr_ActCond[i, sess, j] = actData[actData['phase']=='phase2']['yellowCorrect'].mean()
                        j +=1
                        
                        # Phase 3
                        actPropopr_ActCond[i, sess, j] = actData[actData['phase']=='phase3']['pushCorrect'].mean()
                        clrPropopr_ActCond[i, sess, j] = actData[actData['phase']=='phase3']['yellowCorrect'].mean()
                        j +=1
                elif cond=='Stim':
                    # take data from specific session, run for a subject
                    clrData = behAll[(behAll['session']==sess+1) & (behAll['run']==run+1) & (behAll['block']==cond) & (behAll['sub_ID']==subName)]
                    # if the reversal point is 21
                    if clrData['reverse'].unique()==21:
                        
                        # Phase 1
                        clrPropor_ClrtCond[i, sess, k] = clrData[clrData['phase']=='phase1']['yellowCorrect'].mean()
                        actPropor_ClrCond[i, sess, k] = clrData[clrData['phase']=='phase1']['pushCorrect'].mean()
                        k +=1
                        
                        # Phase 2
                        clrPropor_ClrtCond[i, sess, k] = clrData[clrData['phase']=='phase2']['yellowCorrect'].mean()
                        actPropor_ClrCond[i, sess, k] = clrData[clrData['phase']=='phase2']['pushCorrect'].mean()
                        k +=1
                    # if the reversal point is 14
                    elif clrData['reverse'].unique()==14:
                        
                        # Phase 1
                        clrPropor_ClrtCond[i, sess, k] = clrData[clrData['phase']=='phase1']['yellowCorrect'].mean()
                        actPropor_ClrCond[i, sess, k] = clrData[clrData['phase']=='phase1']['pushCorrect'].mean()
                        k +=1
                        
                        # Phase 2
                        clrPropor_ClrtCond[i, sess, k] = clrData[clrData['phase']=='phase2']['yellowCorrect'].mean()
                        actPropor_ClrCond[i, sess, k] = clrData[clrData['phase']=='phase2']['pushCorrect'].mean()
                        k +=1
                        
                        # Phase 3
                        clrPropor_ClrtCond[i, sess, k] = clrData[clrData['phase']=='phase3']['yellowCorrect'].mean()
                        actPropor_ClrCond[i, sess, k] = clrData[clrData['phase']=='phase3']['pushCorrect'].mean()
                        k +=1

# convert to proportion
actPropopr_ActCond[actPropopr_ActCond<.5] = 1-actPropopr_ActCond[actPropopr_ActCond<.5] 
clrPropopr_ActCond[clrPropopr_ActCond<.5] = 1-clrPropopr_ActCond[clrPropopr_ActCond<.5] 
clrPropor_ClrtCond[clrPropor_ClrtCond<.5] = 1-clrPropor_ClrtCond[clrPropor_ClrtCond<.5] 
actPropor_ClrCond[actPropor_ClrCond<.5] = 1-actPropor_ClrCond[actPropor_ClrCond<.5] 
# an exception in data
actPropopr_ActCond[actPropopr_ActCond==1]=0.66666667

# plot
fig, axes = plt.subplots(2, 2, figsize=(15,8))
axes=axes.flatten()
axes[0].scatter(np.repeat(np.linspace(0, 1, 46), 10), actPropor_ClrCond.flatten(), alpha=.5, s=20)
axes[0].set_xlabel('Participant', fontsize=12)
axes[0].set_title('Color value learning condition', fontsize=12)
axes[0].set_ylabel('Prob. of Action', fontsize=12)
axes[0].set_xticks([])
axes[0].hlines(actPropor_ClrCond.mean(), xmin=0, xmax=1, color='red', linestyle='--')
axes[0].set_ylim(.49, .87)

axes[3].scatter(np.repeat(np.linspace(0, 1, 46), 10), clrPropopr_ActCond.flatten(), alpha=.5, s=20)
axes[3].set_xlabel('Participant', fontsize=12)
axes[3].set_title('Action value learning condition', fontsize=12)
axes[3].set_ylabel('Prob. of Color', fontsize=12)
axes[3].set_xticks([])
axes[3].hlines(clrPropopr_ActCond.mean(), xmin=0, xmax=1, color='red', linestyle='--')
axes[3].set_ylim(.49, .8)

axes[1].scatter(np.repeat(np.linspace(0, 1, 46), 10), actPropopr_ActCond.flatten(), alpha=.5, s=20)
axes[1].set_xlabel('Participant', fontsize=12)
axes[1].set_title('Action value learning condition', fontsize=12)
axes[1].set_ylabel('Prob. of Action', fontsize=12)
axes[1].set_xticks([])
axes[1].hlines(actPropopr_ActCond.mean(), xmin=0, xmax=1, color='red', linestyle='--')
axes[1].set_ylim(.49, .87)

axes[2].scatter(np.repeat(np.linspace(0, 1, 46), 10), clrPropor_ClrtCond.flatten(), alpha=.5, s=20)
axes[2].set_xlabel('Participant', fontsize=12)
axes[2].set_title('Color value learning condition', fontsize=12)
axes[2].set_ylabel('Prob. of Color', fontsize=12)
axes[2].set_xticks([])
axes[2].hlines(clrPropor_ClrtCond.mean(), xmin=0, xmax=1, color='red', linestyle='--')
axes[2].set_ylim(.49, .87)

fig.tight_layout()

fig.savefig('/mrhome/amingk/Documents/7TPD/figures/proportion_ActClr.png', dpi=300)