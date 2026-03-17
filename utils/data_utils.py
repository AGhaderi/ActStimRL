import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import glob
from scipy.io import loadmat
from . import plotting, random, config
 

def build_behavior_dataframe(
    data_dir=config.PROJECT_DATA_DIR,
    beh_all_data_dir=config.PROJECT_BEH_ALL_DATA_DIR,
    raw_outfile=config.PROJECT_RAW_BEH_ALL_FILE,
    cleaned_outfile=config.PROJECT_NoNAN_BEH_ALL_FILE
):
    """
    Build the full behavioral dataframe from raw experiment files.

    Parameters
    ----------
    data_dir : str
        Path to directory containing sub-* folders.
    beh_all_data_dir : str
        Directory storing randomGroupLabel.csv.
    raw_outfile : str
        Output filename for the raw assembled dataset.
    cleaned_outfile : str
        Output filename for the cleaned / NoNaN dataset.

    Returns
    -------
    dataAll : pd.DataFrame
        Full raw concatenated dataset (before exclusions).
    NoNanBehAll : pd.DataFrame
        Cleaned dataset after NaN removal and participant exclusions.
    """

    # ---------------------------------------------------------------
    # Read subjects and group labels
    # ---------------------------------------------------------------
    fileSubjects = glob.glob(f'{data_dir}/sub-*')
    randomGroupLabel = pd.read_csv(f'{beh_all_data_dir}/randomGroupLabel.csv')

    dataAll = pd.DataFrame([])

    # ---------------------------------------------------------------
    # Loop over participants
    # ---------------------------------------------------------------
    for filesubject in fileSubjects:
        subName = filesubject.split('/')[-1]
        labelSes1 = int(randomGroupLabel.loc[randomGroupLabel["sub-ID"] == subName, "ses-02"].iloc[0])
        labelSes2 = int(randomGroupLabel.loc[randomGroupLabel["sub-ID"] == subName, "ses-03"].iloc[0])

        for sess in ["ses-02achieva7t", "ses-03achieva7t"]:

            files = glob.glob(f"{data_dir}/{subName}/{sess}/beh/*.txt")

            if len(files) != 0:
                # ---------------------------------------------------
                # Load .txt behavioral data
                # ---------------------------------------------------
                data = pd.DataFrame([])
                for txtfile in files:
                    df = pd.read_csv(txtfile, sep=r"[\t,]", engine="python")
                    data = pd.concat([data, df])

                data["sub_ID"] = subName
                data.loc[data["session"] == 1, "group"] = str(labelSes1)
                data.loc[data["session"] == 2, "group"] = str(labelSes2)

                # Patient/Control
                data["patient"] = "HC" if labelSes1 == 2 else "PD"

                # Medication decoding
                data["medication"] = data["group"].replace(["1", "2", "3"], ["OFF", "OFF", "ON"])

                # ---------------------------------------------------
                # Load reversal points from .mat files
                # ---------------------------------------------------
                matfiles = glob.glob(f"{data_dir}/{subName}/{sess}/beh/*_beh.mat")
                data_mat = loadmat(matfiles[0])

                blockList1_1 = data_mat["blockList1_1"][0][0]
                blockList1_2 = data_mat["blockList1_2"][0][0]
                blockList2_1 = data_mat["blockList2_1"][0][0]
                blockList2_2 = data_mat["blockList2_2"][0][0]

                data_reverse = np.array([
                    blockList1_1, blockList1_2,
                    blockList2_1, blockList2_2
                ])

                # Identify blocks
                blocks = data.groupby(["run"])["block"].unique().to_numpy()
                runs = data["run"].unique()

                if len(runs) == 2:
                    blocks = np.array([blocks[0][0], blocks[0][1],
                                       blocks[1][0], blocks[1][1]])
                else:
                    blocks = np.array([blocks[0][0], blocks[0][1]])

                # ---------------------------------------------------
                # Assign reversal points and phase labels
                # ---------------------------------------------------
                data["reverse"] = ""
                data["phase"] = ""
                idx = 0

                for run in runs:
                    for b in range(1, 3):
                        data.loc[(data["run"] == run) &
                                 (data["block"] == blocks[idx]), "reverse"] = data_reverse[idx]

                        # Phase assignment
                        subset = data[(data["run"] == run) &
                                      (data["block"] == blocks[idx])]
                        n = subset.shape[0]

                        if n == 27:
                            phase = np.repeat(["phase1", "phase2"], 21)[:27]
                        elif data_reverse[idx] == 21:
                            phase = np.repeat(["phase1", "phase2"], 21)
                        else:
                            phase = np.repeat(["phase1", "phase2", "phase3"], 14)

                        data.loc[(data["run"] == run) &
                                 (data["block"] == blocks[idx]), "phase"] = phase

                        idx += 1

                # Append participant data
                dataAll = pd.concat([dataAll, data])

    # ---------------------------------------------------------------
    # Save raw dataset
    # ---------------------------------------------------------------
    dataAll.to_csv(raw_outfile, index=False)

    # ---------------------------------------------------------------
    # Clean data: remove no-response trials
    # ---------------------------------------------------------------
    temp = dataAll["pushed"].fillna(-1).astype(int).to_numpy()
    NoNanBehAll = dataAll[temp >= 0].copy().reset_index(drop=True)

    # ---------------------------------------------------------------
    # Add indicator columns
    # ---------------------------------------------------------------
    for filesubject in fileSubjects:
        subName = filesubject.split("/")[-1]
        for session in [1, 2]:
            for run in [1, 2]:
                for condition in ["Act", "Stim"]:
                    behAll_indicator = NoNanBehAll[
                        (NoNanBehAll["sub_ID"] == subName) &
                        (NoNanBehAll["block"] == condition) &
                        (NoNanBehAll["session"] == session) &
                        (NoNanBehAll["run"] == run)
                    ]
                    NoNanBehAll.loc[
                        (NoNanBehAll["sub_ID"] == subName) &
                        (NoNanBehAll["block"] == condition) &
                        (NoNanBehAll["session"] == session) &
                        (NoNanBehAll["run"] == run),
                        "indicator"
                    ] = np.arange(1, behAll_indicator.shape[0] + 1)

    # ---------------------------------------------------------------
    # Hard-coded participant exclusions
    # ---------------------------------------------------------------
    NoNanBehAll = NoNanBehAll[
        (NoNanBehAll["sub_ID"] != "sub-010") |
        (NoNanBehAll["session"] != 2) |
        (NoNanBehAll["run"] != 2) |
        (NoNanBehAll["block"] != "Act")
    ]

    NoNanBehAll = NoNanBehAll[
        (NoNanBehAll["sub_ID"] != "sub-030") |
        (NoNanBehAll["session"] != 2) |
        (NoNanBehAll["run"] != 1)
    ]

    NoNanBehAll = NoNanBehAll[
        (NoNanBehAll["sub_ID"] != "sub-086") |
        (NoNanBehAll["session"] != 1) |
        (NoNanBehAll["run"] != 2)
    ]

    withdraw_subs = ["sub-057", "sub-076", "sub-083", "sub-091", "sub-106"]
    for w in withdraw_subs:
        NoNanBehAll = NoNanBehAll[NoNanBehAll["sub_ID"] != w]

    # ---------------------------------------------------------------
    # Save cleaned dataset
    # ---------------------------------------------------------------
    NoNanBehAll.to_csv(cleaned_outfile, index=False)

    print('Both raw and cleaned data are created!')


def calRelevantAndIrrelevantHighRewardOptionTrial(
        readBehFile= config.PROJECT_NoNAN_BEH_ALL_FILE,
        save_file_all = config.PROJECT_NoNAN_BEH_REL_IRREL_HIGH_REWARD_OPTION_ALL_FILE,
        save_file_groupby = config.PROJECT_NoNAN_BEH_REL_IRREL_HIGH_REWARD_OPTION_GROUPBY_ALL_FILE,
    ):
    """
    calcaulte choice behavior related to high-reward option in irelevant and irrelevant value condition
    and reversal structure across all participants and all trials.

    Parameters
    ----------
    readBehFile : str
        Directory where `NoNanBehAll.csv` is stored.
    save_path : str
        Full path where the figure should be saved (csv).

    """

    # ===========================================
    # Load full dataset across all participants
    # ===========================================
    behAll = pd.read_csv(f"{readBehFile}")
    # Fix trial numbering  
    behAll["trialNumber_new"] = behAll["trialNumber"].copy()
    # map 44-85 -> 2-3
    mask = behAll["trialNumber"] >= 44
    behAll.loc[mask, "trialNumber_new"] = behAll.loc[mask, "trialNumber"] - 42
    # shift everything down by 1 -> 1-42
    behAll["trialNumber_new"] = behAll["trialNumber_new"] - 1 
    
    # participatns list
    sub_ID = behAll["sub_ID"].unique()
  
    # ===========================================
    # Compute "relevantHighRewardOption" relevant option
    behAll['relevantHighRewardOptionPattern'] = np.nan
    for sub in sub_ID:
        for block in ["Act", "Stim"]:
            for session in [1,2]:
                for run in [1,2]:
                    # mask
                    mask = ((behAll["sub_ID"] == sub)&(behAll["block"] == block)&
                            (behAll["run"] == run)&(behAll["session"] == session))

                    # filter behavioral data
                    behAllCond = behAll.loc[mask].copy()
                    if len(behAllCond)!=0:
                        # extract phases
                        phases = behAllCond['phase'].unique()

                        # create mask
                        mask_phase1 = mask &(behAll["phase"] == 'phase1')
                        mask_phase2 = mask & (behAll["phase"] == 'phase2')
                        mask_phase3 = mask & (behAll["phase"] == 'phase3')
                            
                        if block == 'Act':

                            # test if the higher probability reward is push, pull, push or pull, push, pull
                            mean_phase1 = behAllCond[behAllCond['phase']=='phase1']['pushCorrect'].mean()
                            
                            if mean_phase1>.5: # push, pull,push
                                if len(phases)==2:
                                    behAll.loc[mask_phase1, 'relevantHighRewardOptionPattern'] = behAllCond[behAllCond['phase']=='phase1']['pushed']
                                    behAll.loc[mask_phase2, 'relevantHighRewardOptionPattern'] = behAllCond[behAllCond['phase']=='phase2']['pushed']                        
                                elif len(phases)==3:
                                    behAll.loc[mask_phase1, 'relevantHighRewardOptionPattern'] = behAllCond[behAllCond['phase']=='phase1']['pushed']
                                    behAll.loc[mask_phase2, 'relevantHighRewardOptionPattern'] = behAllCond[behAllCond['phase']=='phase2']['pushed']
                                    behAll.loc[mask_phase3, 'relevantHighRewardOptionPattern'] = behAllCond[behAllCond['phase']=='phase3']['pushed']

                            else: # pull, push,pull, should be reversed
                                if len(phases)==2:
                                    behAll.loc[mask_phase1, 'relevantHighRewardOptionPattern'] = (behAllCond[behAllCond['phase']=='phase1']['pushed']==0).astype(int)
                                    behAll.loc[mask_phase2, 'relevantHighRewardOptionPattern'] = (behAllCond[behAllCond['phase']=='phase2']['pushed']==0).astype(int)                       
                                elif len(phases)==3:
                                    behAll.loc[mask_phase1, 'relevantHighRewardOptionPattern'] = (behAllCond[behAllCond['phase']=='phase1']['pushed']==0).astype(int)   
                                    behAll.loc[mask_phase2, 'relevantHighRewardOptionPattern'] = (behAllCond[behAllCond['phase']=='phase2']['pushed']==0).astype(int)   
                                    behAll.loc[mask_phase3, 'relevantHighRewardOptionPattern'] = (behAllCond[behAllCond['phase']=='phase3']['pushed']==0).astype(int)   

                        elif block == 'Stim':

                            # test if the higher probability reward is push, pull, push or pull,push,pull
                            mean_phase1 = behAllCond[behAllCond['phase']=='phase1']['yellowCorrect'].mean()
                            
                            if mean_phase1>.5: # push, pull,push
                                if len(phases)==2:
                                    behAll.loc[mask_phase1, 'relevantHighRewardOptionPattern'] = behAllCond[behAllCond['phase']=='phase1']['yellowChosen']
                                    behAll.loc[mask_phase2, 'relevantHighRewardOptionPattern'] = behAllCond[behAllCond['phase']=='phase2']['yellowChosen']                        
                                elif len(phases)==3:
                                    behAll.loc[mask_phase1, 'relevantHighRewardOptionPattern'] = behAllCond[behAllCond['phase']=='phase1']['yellowChosen']
                                    behAll.loc[mask_phase2, 'relevantHighRewardOptionPattern'] = behAllCond[behAllCond['phase']=='phase2']['yellowChosen']
                                    behAll.loc[mask_phase3, 'relevantHighRewardOptionPattern'] = behAllCond[behAllCond['phase']=='phase3']['yellowChosen']

                            else: # pull, push,pull, reverse
                                if len(phases)==2:
                                    behAll.loc[mask_phase1, 'relevantHighRewardOptionPattern'] = (behAllCond[behAllCond['phase']=='phase1']['yellowChosen']==0).astype(int)
                                    behAll.loc[mask_phase2, 'relevantHighRewardOptionPattern'] = (behAllCond[behAllCond['phase']=='phase2']['yellowChosen']==0).astype(int)                       
                                elif len(phases)==3:
                                    behAll.loc[mask_phase1, 'relevantHighRewardOptionPattern'] = (behAllCond[behAllCond['phase']=='phase1']['yellowChosen']==0).astype(int)   
                                    behAll.loc[mask_phase2, 'relevantHighRewardOptionPattern'] = (behAllCond[behAllCond['phase']=='phase2']['yellowChosen']==0).astype(int)   
                                    behAll.loc[mask_phase3, 'relevantHighRewardOptionPattern'] = (behAllCond[behAllCond['phase']=='phase3']['yellowChosen']==0).astype(int)   

    # ===========================================
    # Compute "irrelevantHighRewardOption" relevant option
    behAll['irrelevantHighRewardOption'] = np.nan
    for sub in sub_ID:
        for block in ["Act", "Stim"]:
            for session in [1,2]:
                for run in [1,2]:
                    # mask
                    mask = ((behAll["sub_ID"] == sub)&(behAll["block"] == block)&
                            (behAll["run"] == run)&(behAll["session"] == session))
                    # filter behavioral data
                    behAllCond = behAll.loc[mask].copy()
                    if len(behAllCond)!=0:
                        # extract phases
                        phases = behAllCond['phase'].unique()
                        # loop over phases, 2 or 3
                        for phase in phases:
                            # create mask
                            mask_phase = mask & (behAll["phase"] == phase)
                            
                            # action value condition, irrelevant is color
                            if block =='Act':
                                # check if the higher probability reward is yellow or blue (at random)
                                mean_phase = behAllCond[behAllCond['phase']==phase]['yellowCorrect'].mean()
                                
                                if mean_phase>.5: # push, pull,push
                                    behAll.loc[mask_phase, 'irrelevantHighRewardOption'] = behAllCond[behAllCond['phase']==phase]['yellowChosen']
                                else:
                                    behAll.loc[mask_phase, 'irrelevantHighRewardOption'] = (behAllCond[behAllCond['phase']==phase]['yellowChosen']==0).astype(int)                       
                            
                            # color value condition, irrelevant is action
                            elif block =='Stim':
                                # check if the higher probability reward is yellow or blue (at random)
                                mean_phase = behAllCond[behAllCond['phase']==phase]['pushCorrect'].mean()

                                if mean_phase>.5: # push, pull,push
                                    behAll.loc[mask_phase, 'irrelevantHighRewardOption'] = behAllCond[behAllCond['phase']==phase]['pushed']
                                else:
                                    behAll.loc[mask_phase, 'irrelevantHighRewardOption'] = (behAllCond[behAllCond['phase']==phase]['pushed']==0).astype(int)                       


    # Save behAll with relevant and irrelevant high reward options in csv 
    behAll.to_csv(save_file_all, index=False)


    # calculate the choice proportion for each participant
    behAll["adjusted_phase"] = None
    behAll["relevantHighRewardOption"] = None
    for sub in sub_ID:
        for block in ["Act", "Stim"]:
            for session in [1,2]:
                for run in [1,2]:
                    # create mask
                    mask = ((behAll["sub_ID"] == sub) &
                            (behAll["block"] == block) &
                            (behAll["run"] == run) &
                            (behAll["session"] == session))
                    
                    # filter behavioral data
                    behAllCond = behAll.loc[mask].copy()

                    if len(behAllCond) == 0:
                        continue

                    # extract phases
                    nphases = behAllCond['phase'].nunique() 
                    # two phases
                    if nphases==2:
                        # adjust phases
                        behAll.loc[mask & behAll["trialNumber_new"].between(4, 24), "adjusted_phase"] = "adjPhase1"
                        behAll.loc[mask & behAll["trialNumber_new"].between(25, 42), "adjusted_phase"] = "adjPhase2"
                    
                        # proportion of highRewardOption
                        behAll.loc[mask & behAll["trialNumber_new"].between(4, 24), "relevantHighRewardOption"] =  behAll.loc[mask & behAll["trialNumber_new"].between(4, 24), "relevantHighRewardOptionPattern"]
                        behAll.loc[mask & behAll["trialNumber_new"].between(25, 42), "relevantHighRewardOption"] = 1- behAll.loc[mask & behAll["trialNumber_new"].between(25, 42), "relevantHighRewardOptionPattern"]
                        


                    # three phases
                    elif nphases==3:
                        # adjust phases
                        behAll.loc[mask & behAll["trialNumber_new"].between(4, 17), "adjusted_phase"] = "adjPhase1"
                        behAll.loc[mask & behAll["trialNumber_new"].between(18, 31), "adjusted_phase"] = "adjPhase2"
                        behAll.loc[mask & behAll["trialNumber_new"].between(32, 42), "adjusted_phase"] = "adjPhase3"

                        # proportion of highRewardOption
                        behAll.loc[mask & behAll["trialNumber_new"].between(4, 17), "relevantHighRewardOption"] =  behAll.loc[mask & behAll["trialNumber_new"].between(4, 17), "relevantHighRewardOptionPattern"]
                        behAll.loc[mask & behAll["trialNumber_new"].between(18, 31), "relevantHighRewardOption"] = 1- behAll.loc[mask & behAll["trialNumber_new"].between(18, 31), "relevantHighRewardOptionPattern"]
                        behAll.loc[mask & behAll["trialNumber_new"].between(32, 42), "relevantHighRewardOption"] =  behAll.loc[mask & behAll["trialNumber_new"].between(32, 42), "relevantHighRewardOptionPattern"]

    # group average 
    behAll_new = behAll[behAll['adjusted_phase'].notna()].reset_index(drop=True)
    behAll_new_tempt = behAll_new.groupby( ["group", "sub_ID", 'block', 'adjusted_phase'], as_index=False)[["relevantHighRewardOption", "irrelevantHighRewardOption"]].mean()
    behAll_new_groypby = behAll_new_tempt.groupby( ["group", "sub_ID", 'block'], as_index=False)[["relevantHighRewardOption", "irrelevantHighRewardOption"]].mean()
    
    # Save groupby behAll with relevant and irrelevant high reward options in csv file
    behAll_new_groypby.to_csv(save_file_groupby, index=False)
