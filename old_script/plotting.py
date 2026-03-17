import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.lines as mlines
import glob
import os
from scipy.io import loadmat
from . import config 

def plotChoicePropTrial(
        readBehFile= config.PROJECT_NoNAN_BEH_ALL_FILE,
        save_path = config.FIGURES_DIR,
        window_size: int = 4
    ):
    """
    Plot choice behavior (high-reward option proportion) and reversal structure
    across all participants and all trials.

    Parameters
    ----------
    readBehFile : str
        Directory where `NoNanBehAll.csv` is stored.
    save_path : str
        Full path where the figure should be saved (png).
    window_size : int
        Rolling mean window size for smoothing choice curves.

    Notes
    -----
    - Automatically processes Stim/Act * 1-reversal/2-reversal conditions.
    """

    # ===========================================
    # Load full dataset across all participants
    # ===========================================
    behAll = pd.read_csv(f"{readBehFile}")

    # Fix trial numbering (replace 44-85 -> 2-43)
    behAll["trialNumber"] = behAll["trialNumber"].replace(
                                    list(range(44, 86)),
                                    list(range(2, 44)))
    # participatns
    sub_ID = behAll["sub_ID"].unique()
  
    # ===========================================
    # Loop through conditions
    # ===========================================

    # Compute "highRewardOption" relevant option
    behAll['highRewardOption'] = np.nan
    for sub in sub_ID:
        for block in ["Act", "Stim"]:
            for session in [1,2]:
                for run in [1,2]:
                    # filter behavioral data
                    behAllCond = behAll.loc[(behAll["sub_ID"] == sub)&(behAll["block"] == block)&
                                            (behAll["run"] == run)&(behAll["session"] == session)].copy()
                    if len(behAllCond)!=0:
                        # extract phases
                        phases = behAllCond['phase'].unique()

                        # create mask
                        mask_phase1 = ((behAll["sub_ID"] == sub) &
                                        (behAll["block"] == block) &
                                        (behAll["run"] == run) &
                                        (behAll["session"] == session) &
                                        (behAll["phase"] == 'phase1'))
                        
                        mask_phase2 = ((behAll["sub_ID"] == sub) &
                                        (behAll["block"] == block) &
                                        (behAll["run"] == run) &
                                        (behAll["session"] == session) &
                                        (behAll["phase"] == 'phase2'))
                        
                        mask_phase3 = ((behAll["sub_ID"] == sub) &
                                        (behAll["block"] == block) &
                                        (behAll["run"] == run) &
                                        (behAll["session"] == session) &
                                        (behAll["phase"] == 'phase3'))
                            
                        if block == 'Act':

                            # test if the higher probability reward is push, pull, push or pull, push, pull
                            mean_phase1 = behAllCond[behAllCond['phase']=='phase1']['pushCorrect'].mean()
                            
                            if mean_phase1>.5: # push, pull,push
                                if len(phases)==2:
                                    behAll.loc[mask_phase1, 'highRewardOption'] = behAllCond[behAllCond['phase']=='phase1']['pushed']
                                    behAll.loc[mask_phase2, 'highRewardOption'] = behAllCond[behAllCond['phase']=='phase2']['pushed']                        
                                if len(phases)==3:
                                    behAll.loc[mask_phase1, 'highRewardOption'] = behAllCond[behAllCond['phase']=='phase1']['pushed']
                                    behAll.loc[mask_phase2, 'highRewardOption'] = behAllCond[behAllCond['phase']=='phase2']['pushed']
                                    behAll.loc[mask_phase3, 'highRewardOption'] = behAllCond[behAllCond['phase']=='phase3']['pushed']

                            else: # pull, push,pull, should be reserved
                                if len(phases)==2:
                                    behAll.loc[mask_phase1, 'highRewardOption'] = (behAllCond[behAllCond['phase']=='phase1']['pushed']==0).astype(int)
                                    behAll.loc[mask_phase2, 'highRewardOption'] = (behAllCond[behAllCond['phase']=='phase2']['pushed']==0).astype(int)                       
                                if len(phases)==3:
                                    behAll.loc[mask_phase1, 'highRewardOption'] = (behAllCond[behAllCond['phase']=='phase1']['pushed']==0).astype(int)   
                                    behAll.loc[mask_phase2, 'highRewardOption'] = (behAllCond[behAllCond['phase']=='phase2']['pushed']==0).astype(int)   
                                    behAll.loc[mask_phase3, 'highRewardOption'] = (behAllCond[behAllCond['phase']=='phase3']['pushed']==0).astype(int)   

                        if block == 'Stim':

                            # test if the higher probability reward is push, pull, push or pull,push,pull
                            mean_phase1 = behAllCond[behAllCond['phase']=='phase1']['yellowCorrect'].mean()
                            
                            if mean_phase1>.5: # push, pull,push
                                if len(phases)==2:
                                    behAll.loc[mask_phase1, 'highRewardOption'] = behAllCond[behAllCond['phase']=='phase1']['yellowChosen']
                                    behAll.loc[mask_phase2, 'highRewardOption'] = behAllCond[behAllCond['phase']=='phase2']['yellowChosen']                        
                                if len(phases)==3:
                                    behAll.loc[mask_phase1, 'highRewardOption'] = behAllCond[behAllCond['phase']=='phase1']['yellowChosen']
                                    behAll.loc[mask_phase2, 'highRewardOption'] = behAllCond[behAllCond['phase']=='phase2']['yellowChosen']
                                    behAll.loc[mask_phase3, 'highRewardOption'] = behAllCond[behAllCond['phase']=='phase3']['yellowChosen']

                            else: # pull, push,pull, reverse
                                if len(phases)==2:
                                    behAll.loc[mask_phase1, 'highRewardOption'] = (behAllCond[behAllCond['phase']=='phase1']['yellowChosen']==0).astype(int)
                                    behAll.loc[mask_phase2, 'highRewardOption'] = (behAllCond[behAllCond['phase']=='phase2']['yellowChosen']==0).astype(int)                       
                                if len(phases)==3:
                                    behAll.loc[mask_phase1, 'highRewardOption'] = (behAllCond[behAllCond['phase']=='phase1']['yellowChosen']==0).astype(int)   
                                    behAll.loc[mask_phase2, 'highRewardOption'] = (behAllCond[behAllCond['phase']=='phase2']['yellowChosen']==0).astype(int)   
                                    behAll.loc[mask_phase3, 'highRewardOption'] = (behAllCond[behAllCond['phase']=='phase3']['yellowChosen']==0).astype(int)   

 
    # ===========================================
    # Set up figure
    # ===========================================
    
    mm = 1 / 2.54
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(21*mm, 16*mm))
    axs = axs.flatten()
    idx = 0
    # ===========================================
    for block in ["Act", "Stim"]:
        for reverse in [14, 21]:
            behAllCond = behAll.loc[(behAll["block"] == block) &(behAll["reverse"] == reverse)].copy()

            # ==========================================================
            # Compute average across subjects for each group × trial
            # ==========================================================

            beh_group = behAllCond.groupby( ["group", "trialNumber"], as_index=False)["highRewardOption"].mean()

            # Rolling averages for each participant group
            groups = {
                1: ("PD-OFF", config.COLORS['PD-OFF']),
                2: ("HC", config.COLORS['HC']),
                3: ("PD-ON", config.COLORS['PD-ON']),
            }
        
        
            for group_id, (_, color) in groups.items():
                y = beh_group.loc[
                    beh_group["group"] == group_id, "highRewardOption"
                ]
                moving_avg = y.rolling(window=window_size, min_periods=1).mean()
                axs[idx].plot(np.arange(1, 43), moving_avg, color=color, linewidth=3)

            axs[idx].axhline(0.5, linestyle="--", color="black")

            # ==========================================================
            # Add reversal markers
            # ==========================================================
            if reverse == 21:
                axs[idx].axvline(21, color='c', linestyle='--', alpha=.7)
                axs[idx].plot([0, 21], [.75, .75], color='green')
                axs[idx].plot([21, 42], [.25, .25], color='green')
                axs[idx].plot([21, 21], [.75, .25], color='green')

            else:  # 14 + 28 reversals
                axs[idx].axvline(14, color='c', linestyle='--', alpha=.7)
                axs[idx].axvline(28, color='c', linestyle='--', alpha=.7)
                axs[idx].plot([0, 14], [.75, .75], color='green')
                axs[idx].plot([14, 14], [.75, .25], color='green')
                axs[idx].plot([14, 28], [.25, .25], color='green')
                axs[idx].plot([28, 28], [.75, .25], color='green')
                axs[idx].plot([28, 42], [.75, .75], color='green')

            # Axis configs
            axs[idx].set_ylim(0, 1)
            axs[idx].set_xlim(1, 42)
            axs[idx].set_xticks([1, 10, 20, 30, 42])

            if idx == 0:
                axs[idx].legend(["PD-OFF", "HC","PD-ON"], fontsize=8)

            idx += 1

    # ===========================================
    # Global labels
    # ===========================================
    fig.supxlabel("Trial")
    fig.supylabel("Choice Proportion")

    fig.tight_layout()
    # Save figure
    plt.savefig(f'{save_path}/ChoicePropTrial.pdf')



def plotChoicePropTria_lOld(
        readBehFile= config.PROJECT_NoNAN_BEH_ALL_FILE,
        save_path = config.FIGURES_DIR,
        window_size: int = 4
    ):
    """
    Plot choice behavior (high-reward option proportion) and reversal structure
    across all participants and all trials.

    Parameters
    ----------
    readBehFile : str
        Directory where `NoNanBehAll.csv` is stored.
    save_path : str
        Full path where the figure should be saved (png).
    window_size : int
        Rolling mean window size for smoothing choice curves.

    Notes
    -----
    - Automatically processes Stim/Act * 1-reversal/2-reversal conditions.
    """

    # ===========================================
    # Load full dataset across all participants
    # ===========================================
    behAll = pd.read_csv(f"{readBehFile}")

    # Fix trial numbering (replace 44–85 → 2–43)
    behAll["trialNumber"] = behAll["trialNumber"].replace(
                                    list(range(44, 86)),
                                    list(range(2, 44)))

    # ===========================================
    # Set up figure
    # ===========================================
    mm = 1 / 2.54
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(21*mm, 16*mm))
    axs = axs.flatten()
    idx = 0

    # ===========================================
    # Loop through conditions
    # ===========================================
    for block in ["Act", "Stim"]:
        for reverse in [14, 21]:

            behAllCond = behAll.loc[
                (behAll["block"] == block) &
                (behAll["reverse"] == reverse)
            ].copy()

            # ==========================================================
            # Compute "highRewardOption" depending on block/reversal
            # ==========================================================
            chosenOption = np.zeros(behAllCond.shape[0])

            if block == "Stim" and reverse == 21:
                chosenOption[behAllCond['stimActFirst']=='Stim'] = -(behAllCond.loc[behAllCond['stimActFirst']=='Stim']['yellowChosen']-1)
                chosenOption[behAllCond['stimActFirst']=='Act'] = behAllCond.loc[behAllCond['stimActFirst']=='Act']['yellowChosen']
                axs[idx].set_title("Color-value")

            elif block == "Act" and reverse == 21:
                chosenOption[behAllCond['stimActFirst']=='Stim'] = -(behAllCond.loc[behAllCond['stimActFirst']=='Stim']['pushed']-1)
                chosenOption[behAllCond['stimActFirst']=='Act'] = behAllCond.loc[behAllCond['stimActFirst']=='Act']['pushed']
                axs[idx].set_title("Action-value")

            elif block == "Stim" and reverse == 14:
                chosenOption[behAllCond['stimActFirst']=='Stim'] = -(behAllCond.loc[behAllCond['stimActFirst']=='Stim']['yellowChosen']-1)
                chosenOption[behAllCond['stimActFirst']=='Act'] = -(behAllCond.loc[behAllCond['stimActFirst']=='Act']['yellowChosen']-1)
                axs[idx].set_title("Color-value")

            elif block == "Act" and reverse == 14:
                chosenOption[behAllCond['stimActFirst']=='Stim'] = behAllCond.loc[behAllCond['stimActFirst']=='Stim']['pushed']
                chosenOption[behAllCond['stimActFirst']=='Act'] = -(behAllCond.loc[behAllCond['stimActFirst']=='Act']['pushed']-1)
                axs[idx].set_title("Action-value")

            behAllCond["highRewardOption"] = chosenOption

            # ==========================================================
            # Compute average across subjects for each group × trial
            # ==========================================================
            beh_group = behAllCond.groupby(
                ["group", "trialNumber"],
                as_index=False
            )["highRewardOption"].mean()

            # Rolling averages for each participant group
            groups = {
                1: ("PD-OFF", config.COLORS['PD-OFF']),
                2: ("HC", config.COLORS['HC']),
                3: ("PD-ON", config.COLORS['PD-ON']),
            }

            for group_id, (_, color) in groups.items():
                y = beh_group.loc[
                    beh_group["group"] == group_id, "highRewardOption"
                ]
                moving_avg = y.rolling(window=window_size, min_periods=1).mean()
                axs[idx].plot(np.arange(1, 43), moving_avg, color=color, linewidth=3)

            axs[idx].axhline(0.5, linestyle="--", color="black")

            # ==========================================================
            # Add reversal markers
            # ==========================================================
            if reverse == 21:
                axs[idx].axvline(21, color='c', linestyle='--', alpha=.7)
                axs[idx].plot([0, 21], [.75, .75], color='green')
                axs[idx].plot([21, 42], [.25, .25], color='green')
                axs[idx].plot([21, 21], [.75, .25], color='green')

            else:  # 14 + 28 reversals
                axs[idx].axvline(14, color='c', linestyle='--', alpha=.7)
                axs[idx].axvline(28, color='c', linestyle='--', alpha=.7)
                axs[idx].plot([0, 14], [.75, .75], color='green')
                axs[idx].plot([14, 14], [.75, .25], color='green')
                axs[idx].plot([14, 28], [.25, .25], color='green')
                axs[idx].plot([28, 28], [.75, .25], color='green')
                axs[idx].plot([28, 42], [.75, .75], color='green')

            # Axis configs
            axs[idx].set_ylim(0, 1)
            axs[idx].set_xlim(1, 42)
            axs[idx].set_xticks([1, 10, 20, 30, 42])

            if idx == 0:
                axs[idx].legend(["PD-OFF", "HC","PD-ON"], fontsize=8)

            idx += 1

    # ===========================================
    # Global labels
    # ===========================================
    fig.supxlabel("Trial")
    fig.supylabel("Choice Proportion")

    fig.tight_layout()
    # Save figure
    plt.savefig(f'{save_path}/ChoicePropTrial.pdf')


def plotChoiceResponseSubjects(readBehDir=config.PROJECT_DATA_DIR):
    """
    Reads raw behavioral .txt files and .mat files for each subject and session,
    concatenates them, extracts block order information, and generates behavior plots.

    Parameters
    ----------
    readBehDir : str
        Base directory containing subject/session folders.
    type: ste
        There are differnt type of choice response plot whihc is the name of funcion

    Returns
    -------
    None
    """
     
    # Loop through all subjects
    fileSubjects = glob.glob(f'{readBehDir}/sub-*')
    for filesubject in fileSubjects:
        subName = filesubject.split('/')[-1]

        # Process the two 7T sessions
        for sess in ['ses-02achieva7t', 'ses-03achieva7t']:

            # ---------------------------------------------------------
            # 1. Find all .txt behavioral files for this subject/session
            # ---------------------------------------------------------
            txt_files = glob.glob(f'{readBehDir}/{subName}/{sess}/beh/*.txt')

            if len(txt_files) == 0:
                # No behavioral files found for this session → skip session
                continue

            # ---------------------------------------------------------
            # 2. Load and concatenate all .txt behavioral files
            # ---------------------------------------------------------
            data = pd.DataFrame([])

            for file_path in txt_files:
                # Read with flexible delimiter (tabs or commas)
                df = pd.read_csv(file_path, sep=r'[\t,]', engine='python')

                # Remove extension to get simple filename (not used later, but kept for clarity)
                filename = os.path.splitext(os.path.basename(file_path))[0]

                # Append to the full dataframe
                data = pd.concat([data, df], ignore_index=True)

            # ---------------------------------------------------------
            # 3. Load trial structure (.mat) file
            # ---------------------------------------------------------
            mat_files = glob.glob(f'{readBehDir}/{subName}/{sess}/beh/*_beh.mat')

            if len(mat_files) == 0:
                # No .mat file → skip plotting
                continue

            data_mat = loadmat(mat_files[0])

            # Extract block structure arrays
            blockList1_1 = data_mat['blockList1_1'][0][0]
            blockList1_2 = data_mat['blockList1_2'][0][0]
            blockList2_1 = data_mat['blockList2_1'][0][0]
            blockList2_2 = data_mat['blockList2_2'][0][0]

            # Combine into a 4-element array
            data_reverse = np.array([
                blockList1_1, blockList1_2,
                blockList2_1, blockList2_2
            ])

            # ---------------------------------------------------------
            # 4. Prepare output directory and filename
            # ---------------------------------------------------------
            analysis_dir = f'{readBehDir}/{subName}/{sess}/beh/analysis/'

            if not os.path.isdir(analysis_dir):
                os.makedirs(analysis_dir)

            saveFile = f'{analysis_dir}/{subName}_{sess}_task-DA_beh.png'

            # ---------------------------------------------------------
            # 5. Generate plot (custom module)
            # ---------------------------------------------------------
            plotChoiceResponse(
                data=data,
                subName=subName,
                reverse=data_reverse,
                saveFile=saveFile
            )

            print(f"✓ Saved plot for {subName} {sess}")


def plotChoiceResponse(data, subName, reverse, saveFile):
    """Plot of chosen and correct response for all runs and sessions"""
    # Figure of behavioral data in two column and four rows
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    rows = 2
    columns = 2
    # Position marker type and colors of Action adn Color Value Learning
    y = [1.2 ,1.1, .6 ,.2, .1] 
    markers = ['v', 'o', 'o' , 'o', '^']
    colorsAct =['#2ca02c','#2ca02c', '#d62728', '#9467bd', '#9467bd']
    colorsClr =['#bcbd22','#bcbd22', '#d62728', '#1f77b4', '#1f77b4']
     
    idx = 0
    for run in range(1, 3):
        # data for run
        dataRun = data[(data['run']==run)]
        # order of action and color condition
        blocks = dataRun['block'].unique()
        for block in blocks:
            fig.add_subplot(rows, columns, idx+1) 
            # Action block
            if block == 'Act':
                # Seperate data taken from a session, run and Action block
                dataCondAct = dataRun[(dataRun['block']==block)]
                # Seperate the index of pushed and pulled responses
                resAct = dataCondAct['pushed'].fillna(-1).astype(int).to_numpy()
                pushed = np.where(resAct==1)[0] + 1
                pulled = np.where(resAct==0)[0] + 1
                noRes  = np.where(resAct < 0)[0] + 1
                # Seperate the index of pushed and pulled correct choices
                corrAct= dataCondAct['pushCorrect']
                pushCorr = np.where(corrAct==1)[0] + 1
                pulledCorr = np.where(corrAct==0)[0] + 1
                # Put all reponses and correct choice in a Dataframe
                dicDataAct = ({'label': ['pushed', 'push correct', 'no response', 'pull correct', 'pulled'],
                            'x': [pushed, pushCorr, noRes, pulledCorr, pulled]})
                dfPlotAct = pd.DataFrame(dicDataAct)
                # Create a list of y coordinates for every x coordinate
                for i in range(len(dfPlotAct)):
                    plt.scatter(dfPlotAct.x[i],[y[i] for j in range(len(dfPlotAct.x[i]))], 
                                s=10, c=colorsAct[i], marker=markers[i])
                # show the empy y axis label
                plt.yticks(y,[]) 
                plt.xlabel('Trials', fontsize=12)
                if block=='Stim':
                    blockName = 'Clr'
                elif block=='Act':
                    blockName = 'Act'
                    
                plt.title(subName + ' - Run ' + str(run) + ' - ' +  blockName + ' Value Learning' , fontsize=10)    
                plt.legend(dfPlotAct.label, fontsize=8)      
            # Color block
            elif block == 'Stim':
                # Seperate data taken from a session, run and Color block
                dataCondClr = data[(data.run==run) & (data.block==block)]
                # Seperate the index of yellow and blue responses
                resClr = dataCondClr['yellowChosen'].fillna(-1).astype(int).to_numpy()
                yellChosen = np.where(resClr==1)[0] + 1
                blueChosen = np.where(resClr==0)[0] + 1
                noRes  = np.where(resClr < 0)[0] + 1
                # Seperate the index of yellow and blue correct choices
                corrClr= dataCondClr['yellowCorrect']
                yellCorr = np.where(corrClr==1)[0] + 1
                blueCorr = np.where(corrClr==0)[0] + 1
                # Put all reponses and correct choice in a Dataframe
                dicDataClr = ({'label': ['yellow chosen', 'yellow correct', 'no response', 'blue correct', 'blue chosen'],
                            'x': [yellChosen, yellCorr, noRes, blueCorr, blueChosen]})
                dfPlotClr = pd.DataFrame(dicDataClr)         
                #create a list of y coordinates for every x coordinate
                for i in range(len(dfPlotClr)):
                    plt.scatter(dfPlotClr.x[i],[y[i] for j in range(len(dfPlotClr.x[i]))], 
                                s=10, c=colorsClr[i], marker=markers[i])
                # Show the empy y axis label
                plt.yticks(y,[]) 
                plt.xlabel('Trials', fontsize=12) 
                plt.xlabel('Trials', fontsize=12)
                if block =='Stim':
                    blockName = 'Clr'
                elif block =='Act':
                    blockName = 'Act'
                    
                plt.title(subName + ' - Run ' + str(run) + ' - ' +  blockName + ' Value Learning' , fontsize=10)    
                plt.legend(dfPlotClr.label, fontsize=8)  
            
            
            plt.xlim(0,43)
            # Draw vertical lines for one or two reversal points learning during runs
            if reverse[idx]==21:
                plt.axvline(x = 21, color='#ff7f0e', linewidth=1, alpha=.5)
            elif reverse[idx]==14:
                plt.axvline(x = 14, color='#ff7f0e', linewidth=1, alpha=.7)
                plt.axvline(x = 28, color='#ff7f0e', linewidth=1, alpha=.7)

            idx += 1
 
    # Save plot of chosen and correct response 
    fig.savefig(saveFile, dpi=500)
    plt.close()

     
     
def plotFeatureBias(
    readBehFile=config.PROJECT_NoNAN_BEH_ALL_FILE,
    saveFigPath=config.FIGURES_DIR):
    """
    Load behavioral data, compute response tendencies for different features, 
    and plot probability of choosing each feature across groups and conditions.

    Parameters
    ----------
    readBehFile : str
        Path to the CSV file containing behavioral data.
    saveFigPath : str
        Path where the generated figure will be saved.
    """
    
    # ------------------- Load data -------------------
    behAll = pd.read_csv(readBehFile)
    
    # ------------------- Rearrange trial numbers -------------------
    behAll['trialNumber'] = behAll['trialNumber'].replace(
        list(range(44, 86)),  # old trial numbers
        list(range(2, 44)))   # new trial numbers
    
    # ------------------- Compute chosen amounts and high amount selection -------------------
    chosenAmount = behAll['leftChosen']*behAll['winAmtLeft'] + (1-behAll['leftChosen'])*behAll['winAmtRight'] 
    behAll['chosenHighWinAmt'] = chosenAmount >= 50
    
    # ------------------- Standardize group and condition labels -------------------
    behAll['group'] = behAll['group'].replace([1,2,3], ['PD-OFF', 'HC', 'PD-ON'])
    behAll['Condition'] = behAll['block'].replace(['Act', 'Stim'], ['Action', 'Color'])
    
    # ------------------- Aggregate response tendencies by participant -------------------
    left_groups = behAll.groupby(['group', 'Condition', 'sub_ID'], as_index=False)['leftChosen'].mean()
    amt_groups = behAll.groupby(['group', 'Condition', 'sub_ID'], as_index=False)['chosenHighWinAmt'].mean()
    pushed_groups = behAll.groupby(['group', 'Condition', 'sub_ID'], as_index=False)['pushed'].mean()
    yellow_groups = behAll.groupby(['group', 'Condition', 'sub_ID'], as_index=False)['yellowChosen'].mean()
    
    # -------------------Plot responses -------------------
    mm = 1/2.54  # convert cm to inches for figure size
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(21*mm, 16*mm))
    axs = axs.flatten()
    
    # Custom color palette
    custom_palette = {'HC': config.COLORS['HC'], 'PD-ON': config.COLORS['PD-ON'], 'PD-OFF': config.COLORS['PD-OFF']}
    
    # ------------------- Left responses -------------------
    sns.boxplot(data=left_groups, x='Condition', y='leftChosen', hue='group', ax=axs[0],
                palette=custom_palette, showfliers=False)
    sns.stripplot(data=left_groups, x='Condition', y='leftChosen', hue='group', ax=axs[0],
                  dodge=True, alpha=1, size=4, legend=False, palette='dark:black')
    axs[0].set_ylabel('Left response', fontsize=12)
    axs[0].set_xlabel('', fontsize=12)
    axs[0].axhline(.5, color='black', linestyle='--')
    axs[0].set_ylim(0,1)
    axs[0].legend(fontsize=8, loc='upper left')
    
    # ------------------- High amount responses -------------------
    sns.boxplot(data=amt_groups, x='Condition', y='chosenHighWinAmt', hue='group', ax=axs[1],
                palette=custom_palette, showfliers=False,legend=False)
    sns.stripplot(data=amt_groups, x='Condition', y='chosenHighWinAmt', hue='group', ax=axs[1],
                  dodge=True, alpha=1, size=4, legend=False, palette='dark:black')
    axs[1].set_ylabel('Higher amount', fontsize=12)
    axs[1].set_xlabel('', fontsize=12)
    axs[1].axhline(.5, color='black', linestyle='--')
    axs[1].set_ylim(0,1)
    
    # ------------------- Push responses -------------------
    sns.boxplot(data=pushed_groups, x='Condition', y='pushed', hue='group', ax=axs[2],
                palette=custom_palette, showfliers=False,legend=False)
    sns.stripplot(data=pushed_groups, x='Condition', y='pushed', hue='group', ax=axs[2],
                  dodge=True, alpha=1, size=4, legend=False, palette='dark:black')
    axs[2].set_ylabel('Push response', fontsize=12)
    axs[2].set_xlabel('Condition', fontsize=12)
    axs[2].axhline(.5, color='black', linestyle='--')
    axs[2].set_ylim(0,1)
    
    # ------------------- Yellow responses -------------------
    sns.boxplot(data=yellow_groups, x='Condition', y='yellowChosen', hue='group', ax=axs[3],
                palette=custom_palette, showfliers=False,legend=False)
    sns.stripplot(data=yellow_groups, x='Condition', y='yellowChosen', hue='group', ax=axs[3],
                  dodge=True, alpha=1, size=4, legend=False, palette='dark:black')
    axs[3].set_ylabel('Yellow response', fontsize=12)
    axs[3].set_xlabel('Condition', fontsize=12)
    axs[3].axhline(.5, color='black', linestyle='--')
    axs[3].set_ylim(0,1)
    
    fig.tight_layout()
    
    # ------------------- 8. Save figure -------------------
    plt.savefig(f'{saveFigPath}/featureBias.pdf')
    plt.show()


# Taken from https://github.com/laurafontanesi/rlssm/blob/main/rlssm/utils.py 
def bci(x, alpha=0.05):
    """Calculate Bayesian credible interval (BCI).
    Parameters
    ----------
    x : array-like
        An array containing MCMC samples.
    alpha : float
        Desired probability of type I error (defaults to 0.05).
    Returns
    -------
    interval : numpy.ndarray
        Array containing the lower and upper bounds of the bci interval.
    """

    interval = np.nanpercentile(x, [(alpha/2)*100, (1-alpha/2)*100])

    return interval

def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of a given width.
    Parameters
    ----------
    x : array-like
        An sorted numpy array.
    alpha : float
        Desired probability of type I error (defaults to 0.05).
    Returns
    -------
    hdi_min : float
        The lower bound of the interval.
    hdi_max : float
        The upper bound of the interval.
    """

    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max


def hdi(x, alpha=0.05):
    """Calculate highest posterior density (HPD).
        Parameters
        ----------
        x : array-like
            An array containing MCMC samples.
        alpha : float
            Desired probability of type I error (defaults to 0.05).
    Returns
    -------
    interval : numpy.ndarray
        Array containing the lower and upper bounds of the hdi interval.
    """

    # Make a copy of trace
    x = x.copy()
     # Sort univariate node
    sx = np.sort(x)
    interval = np.array(calc_min_interval(sx, alpha))

    return interval

def plot_posterior(x,
                   ax=None,
                   gridsize=100,
                   clip=None,
                   show_intervals="HDI",
                   alpha_intervals=.05,
                   color='grey',
                   intervals_kws=None,
                   trueValue = None,
                   title = None,
                   xlabel = None,
                   ylabel = None,
                   legends = None,
                   **kwargs):
    """Plots a univariate distribution with Bayesian intervals for inference.

    By default, only plots the kernel density estimation using scipy.stats.gaussian_kde.

    Bayesian instervals can be also shown as shaded areas,
    by changing show_intervals to either BCI or HDI.

    Parameters
    ----------

    x : array-like
        Usually samples from a posterior distribution.

    ax : matplotlib.axes.Axes, optional
        If provided, plot on this Axes.
        Default is set to current Axes.

    gridsize : int, default to 100
        Resolution of the kernel density estimation function.

    clip : tuple of (float, float), optional
        Range for the kernel density estimation function.
        Default is min and max values of `x`.

    show_intervals : str, default to "HDI"
        Either "HDI", "BCI", or None.
        HDI is better when the distribution is not simmetrical.
        If None, then no intervals are shown.

    alpha_intervals : float, default to .05
        Alpha level for the intervals calculation.
        Default is 5 percent which gives 95 percent BCIs and HDIs.

    intervals_kws : dict, optional
        Additional arguments for `matplotlib.axes.Axes.fill_between`
        that shows shaded intervals.
        By default, they are 50 percent transparent.

    color : matplotlib.colors
        Color for both the density curve and the intervals.

    Returns
    -------

    ax : matplotlib.axes.Axes
        Returns the `matplotlib.axes.Axes` object with the plot
        for further tweaking.

    """
    if clip is None:
        min_x = np.min(x)
        max_x = np.max(x)
    else:
        min_x, max_x = clip

    if ax is None:
        ax = plt.gca()
        
    if trueValue is not None:
        ax.axvline(x=trueValue, ls='--')

    if intervals_kws is None:
        intervals_kws = {'alpha':.5}

    density = gaussian_kde(x, bw_method='scott')
    xd = np.linspace(min_x, max_x, gridsize)
    yd = density(xd)

    ax.plot(xd, yd, color=color, **kwargs)

    if show_intervals is not None:
        if np.sum(show_intervals == np.array(['BCI', 'HDI'])) < 1:
            raise ValueError("must be either None, BCI, or HDI")
        if show_intervals == 'BCI':
            low, high = bci(x, alpha_intervals)
        else:
            low, high = hdi(x, alpha_intervals)
        ax.fill_between(xd[np.logical_and(xd >= low, xd <= high)],
                        yd[np.logical_and(xd >= low, xd <= high)],
                        color=color,
                        **intervals_kws)
    
    if legends is not None:
        ax.legend(legends) 
        
    if title is not None:
        plt.title(title, fontsize=12)

    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=12)
    
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=14)
    
   
    sns.despine()
     
    return ax    



