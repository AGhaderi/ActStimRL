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



