import numpy as np
import pickle
import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from utils import *
from scipy.stats import kurtosis
import json 
from scipy import stats


def compute_and_save_clinical_parameters(
    readClicalEvalFile=PROJECT_CLIN_EVAL_FILE,
    readModel=SCRATCH_HIER_MODEL_DIR,
    outDir=SCRATCH_CLIN_EVAL_DIR,
    outFile=SCARTCH_CLIN_EVAL_FILE
):
    """
    Computes MAP estimates (via KDE mode) of hierarchical RL parameters for PD & HC,
    merges them with clinical evaluation data, computes medication effects,
    and saves the combined table to CSV.

    Parameters
    ----------
    readClicalEvalDIR : str
        Directory containing behavioral and model data (pickle files, clinical_evaluation.csv).

    outDir : str
        Directory where the final output CSV will be written.

    Returns
    -------
    parameter_clinical_evaluation : pd.DataFrame
        Table containing clinical + model-derived parameters.
    """

    
    # Helper function: KDE mode
    def get_mode_density(values):
        """Return the mode of a posterior distribution using KDE."""
        kde = gaussian_kde(values)
        x_grid = np.linspace(min(values), max(values), 1000)
        return x_grid[np.argmax(kde(x_grid))]

    # read collected data across all participants
    behAll = pd.read_csv(PROJECT_NoNAN_BEH_ALL_FILE)
    # select group 
    behAll_PD = behAll[(behAll['patient']=='PD')].copy().reset_index(drop=False)
    behAll_HC = behAll[(behAll['patient']=='HC')].copy().reset_index(drop=False)
    #  participant
    particiapnts_PD = behAll_PD['sub_ID'].unique()
    particiapnts_HC = behAll_HC['sub_ID'].unique()

    # Load clinical evaluation
    clinical_evaluation = pd.read_csv(f'{readClicalEvalFile}')

    # LOAD PD MODEL RESULTS  
    pkl_PD = f'{readModel}/Tabel3/PD/tabel3_model1_complement_prob_PD.pkl'
    fit_PD = load_pickle(load_path=pkl_PD)['fit']

    # Extract posterior samples
    transfer_alpha_pos_PD = fit_PD["transfer_alpha_pos"]
    transfer_alpha_neg_PD = fit_PD["transfer_alpha_neg"]
    transfer_sensitivity_PD = fit_PD["transfer_sensitivity"]
    transfer_weight_PD = fit_PD["transfer_weight"]

    nParts = transfer_alpha_pos_PD.shape[0]
    nMeds = transfer_alpha_pos_PD.shape[1]
    nConds = 2

    # Initialize MAP arrays
    map_alpha_pos_PD = np.zeros((nParts, nMeds))
    map_alpha_neg_PD = np.zeros((nParts, nConds, nMeds))
    map_sensitivity_PD = np.zeros((nParts, nConds, nMeds))
    map_weighting_PD = np.zeros((nParts, nConds, nMeds))

    # Positive LR
    for i in range(nParts):
        for j in range(nMeds):
            map_alpha_pos_PD[i, j] = get_mode_density(transfer_alpha_pos_PD[i, j])

    # Negative LR, sensitivity, weighting
    for i in range(nParts):
        for j in range(nConds):
            for k in range(nMeds):
                map_alpha_neg_PD[i, j, k] = get_mode_density(transfer_alpha_neg_PD[i, j, k])
                map_weighting_PD[i, j, k] = get_mode_density(transfer_weight_PD[i, j, k])
                map_sensitivity_PD[i, j, k] = get_mode_density(transfer_sensitivity_PD[i, j, k])

    # Medication effects (PD ON - PD OFF)
    map_med_alpha_pos_PD = map_alpha_pos_PD[:, 1] - map_alpha_pos_PD[:, 0]
    map_mean_alpha_pos_PD = np.mean([map_alpha_pos_PD[:, 1], map_alpha_pos_PD[:, 0]], axis=0)

    map_med_alpha_neg_PD = np.mean([map_alpha_neg_PD[:, 0, 1], map_alpha_neg_PD[:, 1, 1]], axis=0) - \
                           np.mean([map_alpha_neg_PD[:, 0, 0], map_alpha_neg_PD[:, 1, 0]], axis=0)

    map_mean_alpha_neg_PD = np.mean([
        map_alpha_neg_PD[:, 0, 0], map_alpha_neg_PD[:, 0, 1],
        map_alpha_neg_PD[:, 1, 0], map_alpha_neg_PD[:, 1, 1]
    ], axis=0)

    map_med_sensitivity_PD = np.mean([map_sensitivity_PD[:, 0, 1], map_sensitivity_PD[:, 1, 1]], axis=0) - \
                             np.mean([map_sensitivity_PD[:, 0, 0], map_sensitivity_PD[:, 1, 0]], axis=0)

    map_mean_sensitivity_PD = np.mean([
        map_sensitivity_PD[:, 0, 0], map_sensitivity_PD[:, 0, 1],
        map_sensitivity_PD[:, 1, 0], map_sensitivity_PD[:, 1, 1]
    ], axis=0)

    # Weighting parameter
    map_med_weighting_act_PD = map_weighting_PD[:, 0, 1] - map_weighting_PD[:, 0, 0]
    map_mean_weighting_act_PD = np.mean([map_weighting_PD[:, 0, 1], map_weighting_PD[:, 0, 0]], axis=0)

    map_med_weighting_clr_PD = map_weighting_PD[:, 1, 1] - map_weighting_PD[:, 1, 0]
    map_mean_weighting_clr_PD = np.mean([map_weighting_PD[:, 1, 1], map_weighting_PD[:, 1, 0]], axis=0)

    map_med_weighting_PD = map_med_weighting_act_PD + map_med_weighting_clr_PD
    map_mean_weighting_PD = np.mean([
        map_weighting_PD[:, 0, 1], map_weighting_PD[:, 0, 0],
        map_weighting_PD[:, 1, 1], map_weighting_PD[:, 1, 0]
    ], axis=0)

    
    # LOAD HC MODEL RESULTS
    pkl_HC = f'{readModel}/Tabel3/HC/tabel3_model1_complement_prob_HC.pkl'
    fit_HC = load_pickle(load_path=pkl_HC)['fit']

    transfer_alpha_pos_HC = fit_HC["transfer_alpha_pos"]
    transfer_alpha_neg_HC = fit_HC["transfer_alpha_neg"]
    transfer_sensitivity_HC = fit_HC["transfer_sensitivity"]
    transfer_weight_HC = fit_HC["transfer_weight"]

    nParts = transfer_alpha_pos_HC.shape[0]

    map_alpha_pos_HC = np.zeros((nParts, 2))
    map_alpha_neg_HC = np.zeros((nParts, 2, 2))
    map_sensitivity_HC = np.zeros((nParts, 2, 2))
    map_weighting_HC = np.zeros((nParts, 2, 2))

    for i in range(nParts):
        for j in range(2):
            map_alpha_pos_HC[i, j] = get_mode_density(transfer_alpha_pos_HC[i, j])

    for i in range(nParts):
        for j in range(2):
            for k in range(2):
                map_alpha_neg_HC[i, j, k] = get_mode_density(transfer_alpha_neg_HC[i, j, k])
                map_weighting_HC[i, j, k] = get_mode_density(transfer_weight_HC[i, j, k])
                map_sensitivity_HC[i, j, k] = get_mode_density(transfer_sensitivity_HC[i, j, k])

    map_mean_alpha_pos_HC = np.mean([map_alpha_pos_HC[:, 1], map_alpha_pos_HC[:, 0]], axis=0)

    map_mean_alpha_neg_HC = np.mean([
        map_alpha_neg_HC[:, 0, 0], map_alpha_neg_HC[:, 0, 1],
        map_alpha_neg_HC[:, 1, 0], map_alpha_neg_HC[:, 1, 1]
    ], axis=0)

    map_mean_sensitivity_HC = np.mean([
        map_sensitivity_HC[:, 0, 0], map_sensitivity_HC[:, 0, 1],
        map_sensitivity_HC[:, 1, 0], map_sensitivity_HC[:, 1, 1]
    ], axis=0)

    map_mean_weighting_act_HC = np.mean([map_weighting_HC[:, 0, 1], map_weighting_HC[:, 0, 0]], axis=0)
    map_mean_weighting_clr_HC = np.mean([map_weighting_HC[:, 1, 1], map_weighting_HC[:, 1, 0]], axis=0)

    map_mean_weighting_HC = np.mean([
        map_weighting_HC[:, 0, 1], map_weighting_HC[:, 0, 0],
        map_weighting_HC[:, 1, 1], map_weighting_HC[:, 1, 0]
    ], axis=0)

    
    #MERGE MODEL PARAMETERS WITH CLINICAL DATA ----
    parameter_clinical_evaluation = clinical_evaluation.copy()

    # Assign parameters in PD
    for sub, subject in enumerate(particiapnts_PD):

        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_mean_alpha_pos'] = map_mean_alpha_pos_PD[sub]

        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_mean_alpha_neg'] = map_mean_alpha_neg_PD[sub]

        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_mean_sensitivity'] = map_mean_sensitivity_PD[sub]

        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_mean_weighting_act'] = map_mean_weighting_act_PD[sub]

        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_mean_weighting_clr'] = map_mean_weighting_clr_PD[sub]

        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_mean_weighting'] = map_mean_weighting_PD[sub]

        # PD-specific medication effects
        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_med_alpha_pos'] = map_med_alpha_pos_PD[sub]
        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_med_alpha_neg'] = map_med_alpha_neg_PD[sub]
        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_med_sensitivity'] = map_med_sensitivity_PD[sub]
        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_med_weighting_act'] = map_med_weighting_act_PD[sub]
        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_med_weighting_clr'] = map_med_weighting_clr_PD[sub]
        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_med_weighting'] = map_med_weighting_PD[sub]

        # UPDRS difference
        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'med_UPDRS'] = \
            parameter_clinical_evaluation['total_UPDRSON'] - parameter_clinical_evaluation['total_UPDRSOFF']


    for sub, subject in enumerate(particiapnts_HC):
        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_mean_alpha_pos'] = map_mean_alpha_pos_HC[sub]

        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_mean_alpha_pos'] = map_mean_alpha_pos_HC[sub]

        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_mean_alpha_neg'] = map_mean_alpha_neg_HC[sub]

        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_mean_sensitivity'] = map_mean_sensitivity_HC[sub]

        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_mean_weighting_act'] = map_mean_weighting_act_HC[sub]

        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_mean_weighting_clr'] = map_mean_weighting_clr_HC[sub]

        parameter_clinical_evaluation.loc[parameter_clinical_evaluation['sub_ID']==subject, 'map_mean_weighting'] = map_mean_weighting_HC[sub]
    
    # Save CSV

    # Check out if it does not exist
    if not os.path.isdir(f'{outDir}'):
            os.makedirs(f'{outDir}') 

    parameter_clinical_evaluation.to_csv(outFile, index=False)

    print(f"Saved clinical parameter table to:\n{outDir}")


def dataStanActClr(readBehFile= PROJECT_NoNAN_BEH_ALL_FILE, group:str='PD',
                   table:str='table3', model:str='model1'):
    """
    Prepare and standardize behavioral data for Action and Color conditions.
    Converts categorical labels to numeric indices and organizes data into a dictionary
    suitable for modeling or further analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Behavioral data  
        Group to select: 'HC' for healthy controls or 'PD' for Parkinson's patients.

    Returns
    -------
    dataStan : dict
        Dictionary containing standardized data arrays 
    """
    # Load full dataset across all participants
    behAll = pd.read_csv(f"{readBehFile}")

    # Select only participants from the specified group 
    data = behAll[(behAll['patient'] == group)].copy().reset_index(drop=False)

    # Count number of participants 
    nParts = len(data['sub_ID'].unique())

    # Convert participant IDs to consecutive integer indices 
    id_map = dict(zip(data['sub_ID'].unique(), np.arange(1, nParts + 1)))
    data['sub_ID'] = data['sub_ID'].map(id_map).astype(int)

    # Number of conditions 
    nConds = 2  # 1 = Action (Act), 2 = Color (Stim)

    # Convert condition labels to integers 
    # 'Act' -> 1, 'Stim' -> 2
    data['block'] = data['block'].map({'Act': 1, 'Stim': 2}).astype(int)

    # Number of sessions / medication conditions 
    nMeds_nSes = 2

    # Set session or medication variable based on group 
    if group == 'HC':
        # For healthy controls, use session column directly
        medication_session = np.array(data['session']).astype(int)
    elif group == 'PD':
        # For PD patients, map group labels: 1 -> OFF, 3 -> ON
        data['medication'] = data['group'].replace([1, 3], [1, 2]).astype(int)
        medication_session = np.array(data['medication']).astype(int)

    # Organize data into a dictionary 
    # Each key will be used for modeling or analysis (e.g., in Stan or other frameworks)
    dataStan = {
        'N': data.shape[0],  # Total number of trials
        'nParts': nParts,    # Number of participants
        'pushed': np.array(data['pushed']).astype(int),              # Action choice (push=1, pull=0)
        'yellowChosen': np.array(data['yellowChosen']).astype(int),  # Color choice (yellow=1, blue=0)
        'leftChosen': np.array(data['leftChosen']).astype(int),      # Side choice (left=1, right=0)
        'winAmtPushable': np.array(data['winAmtPushable']),          # Reward if pushed
        'winAmtPullable': np.array(data['winAmtPullable']),          # Reward if pulled
        'winAmtYellow': np.array(data['winAmtYellow']),              # Reward if yellow is chosen
        'winAmtBlue': np.array(data['winAmtBlue']),                  # Reward if blue is chosen
        'winAmtLeft': np.array(data['winAmtLeft']),                  # Reward if left is chosen
        'winAmtRight': np.array(data['winAmtRight']),                # Reward if right is chosen
        'rewarded': np.array(data['correctChoice']).astype(int),     # Whether choice is correct
        'participant': np.array(data['sub_ID']).astype(int),         # Participant index
        'indicator': np.array(data['indicator']).astype(int),        # Trial indicator variable
        'cond': np.array(data['block']).astype(int),                 # Condition per trial (1=Act, 2=Clr)
        'medSess': medication_session                                # Session or medication index
    } 

    # adjust the dataStan for each model
    #################### Tabel2
    if model=='model1' and table=='tabel2':
        dataStan['n_conds_alpha_pos']=2;    # Number of conditions for positive learning rate
        dataStan['n_medSess_alpha_pos']=1;  # Number of session/medication for positive learning rate
        dataStan['n_conds_alpha_neg']=2;    # Number of conditions for negative learning rate
        dataStan['n_medSess_alpha_neg']=1;  # Number of session/medication for negative learning rate
        dataStan['n_conds_weight']=2;       # Number of conditions for weighing
        dataStan['n_medSess_weight']=1;     # Number of session/medication for weighting 
        dataStan['n_conds_sensitivity']=2;  # Number of conditions for sensitivity
        dataStan['n_medSess_sensitivity']=1;# Number of session/medication for sensitivity
    
    elif model=='model2' and table=='tabel2':
        dataStan['n_conds_alpha_pos']=1;    
        dataStan['n_medSess_alpha_pos']=1;  
        dataStan['n_conds_alpha_neg']=1;    
        dataStan['n_medSess_alpha_neg']=1;  
        dataStan['n_conds_weight']=2;       
        dataStan['n_medSess_weight']=1;     
        dataStan['n_conds_sensitivity']=2;  
        dataStan['n_medSess_sensitivity']=1;

    elif model=='model3' and table=='tabel2':
        dataStan['n_conds_alpha_pos']=2;    
        dataStan['n_medSess_alpha_pos']=1;  
        dataStan['n_conds_alpha_neg']=2;    
        dataStan['n_medSess_alpha_neg']=1;  
        dataStan['n_conds_weight']=2;       
        dataStan['n_medSess_weight']=1;     
        dataStan['n_conds_sensitivity']=1;  
        dataStan['n_medSess_sensitivity']=1;
    
    if model=='model4' and table=='tabel2':
        dataStan['n_conds_alpha_pos']=2;     
        dataStan['n_medSess_alpha_pos']=1;   
        dataStan['n_conds_alpha_neg']=1;     
        dataStan['n_medSess_alpha_neg']=1;   
        dataStan['n_conds_weight']=2;        
        dataStan['n_medSess_weight']=1;     
        dataStan['n_conds_sensitivity']=2;   
        dataStan['n_medSess_sensitivity']=1; 
    
    if model=='model5' and table=='tabel2':
        dataStan['n_conds_alpha_pos']=1;     
        dataStan['n_medSess_alpha_pos']=1;   
        dataStan['n_conds_alpha_neg']=2;     
        dataStan['n_medSess_alpha_neg']=1;   
        dataStan['n_conds_weight']=2;        
        dataStan['n_medSess_weight']=1;     
        dataStan['n_conds_sensitivity']=2;   
        dataStan['n_medSess_sensitivity']=1; 
    
    if model=='model6' and table=='tabel2':
        dataStan['n_conds_alpha_pos']=1;     
        dataStan['n_medSess_alpha_pos']=1;   
        dataStan['n_conds_alpha_neg']=1;     
        dataStan['n_medSess_alpha_neg']=1;   
        dataStan['n_conds_weight']=2;        
        dataStan['n_medSess_weight']=1;     
        dataStan['n_conds_sensitivity']=1;   
        dataStan['n_medSess_sensitivity']=1; 

    #################### Tabel3
    elif model=='model1' and table=='tabel3' and group=='HC':
        dataStan['n_conds_alpha_pos']=2;    
        dataStan['n_medSess_alpha_pos']=2;  
        dataStan['n_conds_alpha_neg']=2;    
        dataStan['n_medSess_alpha_neg']=2;  
        dataStan['n_conds_weight']=2;       
        dataStan['n_medSess_weight']=2;     
        dataStan['n_conds_sensitivity']=2;  
        dataStan['n_medSess_sensitivity']=2;
    elif model=='model2' and table=='tabel3' and group=='HC':
        dataStan['n_conds_alpha_pos']=2;    
        dataStan['n_medSess_alpha_pos']=2;  
        dataStan['n_conds_alpha_neg']=2;    
        dataStan['n_medSess_alpha_neg']=2;  
        dataStan['n_conds_weight']=2;       
        dataStan['n_medSess_weight']=1;     
        dataStan['n_conds_sensitivity']=2;  
        dataStan['n_medSess_sensitivity']=2;
    elif model=='model3' and table=='tabel3' and group=='HC':
        dataStan['n_conds_alpha_pos']=2;    
        dataStan['n_medSess_alpha_pos']=1;  
        dataStan['n_conds_alpha_neg']=2;    
        dataStan['n_medSess_alpha_neg']=1;  
        dataStan['n_conds_weight']=2;       
        dataStan['n_medSess_weight']=2;     
        dataStan['n_conds_sensitivity']=2;  
        dataStan['n_medSess_sensitivity']=2;
    elif model=='model4' and table=='tabel3' and group=='HC':
        dataStan['n_conds_alpha_pos']=2;    
        dataStan['n_medSess_alpha_pos']=2;  
        dataStan['n_conds_alpha_neg']=2;    
        dataStan['n_medSess_alpha_neg']=2;  
        dataStan['n_conds_weight']=2;       
        dataStan['n_medSess_weight']=2;     
        dataStan['n_conds_sensitivity']=2;  
        dataStan['n_medSess_sensitivity']=1;
    elif model=='model5' and table=='tabel3' and group=='HC':
        dataStan['n_conds_alpha_pos']=2;    
        dataStan['n_medSess_alpha_pos']=1;  
        dataStan['n_conds_alpha_neg']=2;    
        dataStan['n_medSess_alpha_neg']=1;  
        dataStan['n_conds_weight']=2;       
        dataStan['n_medSess_weight']=2;     
        dataStan['n_conds_sensitivity']=2;  
        dataStan['n_medSess_sensitivity']=1;
    elif model=='model6' and table=='tabel3' and group=='HC':
        dataStan['n_conds_alpha_pos']=2;    
        dataStan['n_medSess_alpha_pos']=2;  
        dataStan['n_conds_alpha_neg']=2;    
        dataStan['n_medSess_alpha_neg']=1;  
        dataStan['n_conds_weight']=2;       
        dataStan['n_medSess_weight']=2;     
        dataStan['n_conds_sensitivity']=2;  
        dataStan['n_medSess_sensitivity']=2;
    elif model=='model7' and table=='tabel3' and group=='HC':
        dataStan['n_conds_alpha_pos']=2;    
        dataStan['n_medSess_alpha_pos']=1;  
        dataStan['n_conds_alpha_neg']=2;    
        dataStan['n_medSess_alpha_neg']=2;  
        dataStan['n_conds_weight']=2;       
        dataStan['n_medSess_weight']=2;     
        dataStan['n_conds_sensitivity']=2;  
        dataStan['n_medSess_sensitivity']=2;
    elif model=='model8' and table=='tabel3' and group=='HC':
        dataStan['n_conds_alpha_pos']=2;    
        dataStan['n_medSess_alpha_pos']=1;  
        dataStan['n_conds_alpha_neg']=2;    
        dataStan['n_medSess_alpha_neg']=1;  
        dataStan['n_conds_weight']=2;       
        dataStan['n_medSess_weight']=1;     
        dataStan['n_conds_sensitivity']=2;  
        dataStan['n_medSess_sensitivity']=1;


    elif model=='model1' and table=='tabel3' and group=='PD':
        dataStan['n_conds_alpha_pos']=1;    
        dataStan['n_medSess_alpha_pos']=2;  
        dataStan['n_conds_alpha_neg']=2;    
        dataStan['n_medSess_alpha_neg']=2;  
        dataStan['n_conds_weight']=2;       
        dataStan['n_medSess_weight']=2;     
        dataStan['n_conds_sensitivity']=2;  
        dataStan['n_medSess_sensitivity']=2;
    elif model=='model2' and table=='tabel3' and group=='PD':
        dataStan['n_conds_alpha_pos']=1;    
        dataStan['n_medSess_alpha_pos']=2;  
        dataStan['n_conds_alpha_neg']=2;    
        dataStan['n_medSess_alpha_neg']=2;  
        dataStan['n_conds_weight']=2;       
        dataStan['n_medSess_weight']=1;     
        dataStan['n_conds_sensitivity']=2;  
        dataStan['n_medSess_sensitivity']=2;
    elif model=='model3' and table=='tabel3' and group=='PD':
        dataStan['n_conds_alpha_pos']=1;    
        dataStan['n_medSess_alpha_pos']=1;  
        dataStan['n_conds_alpha_neg']=2;    
        dataStan['n_medSess_alpha_neg']=1;  
        dataStan['n_conds_weight']=2;       
        dataStan['n_medSess_weight']=2;     
        dataStan['n_conds_sensitivity']=2;  
        dataStan['n_medSess_sensitivity']=2;
    elif model=='model4' and table=='tabel3' and group=='PD':
        dataStan['n_conds_alpha_pos']=1;    
        dataStan['n_medSess_alpha_pos']=2;  
        dataStan['n_conds_alpha_neg']=2;    
        dataStan['n_medSess_alpha_neg']=2;  
        dataStan['n_conds_weight']=2;       
        dataStan['n_medSess_weight']=2;     
        dataStan['n_conds_sensitivity']=2;  
        dataStan['n_medSess_sensitivity']=1;
    elif model=='model5' and table=='tabel3' and group=='PD':
        dataStan['n_conds_alpha_pos']=1;    
        dataStan['n_medSess_alpha_pos']=1;  
        dataStan['n_conds_alpha_neg']=2;    
        dataStan['n_medSess_alpha_neg']=1;  
        dataStan['n_conds_weight']=2;       
        dataStan['n_medSess_weight']=2;     
        dataStan['n_conds_sensitivity']=2;  
        dataStan['n_medSess_sensitivity']=1;
    elif model=='model6' and table=='tabel3' and group=='PD':
        dataStan['n_conds_alpha_pos']=1;    
        dataStan['n_medSess_alpha_pos']=2;  
        dataStan['n_conds_alpha_neg']=2;    
        dataStan['n_medSess_alpha_neg']=1;  
        dataStan['n_conds_weight']=2;       
        dataStan['n_medSess_weight']=2;     
        dataStan['n_conds_sensitivity']=2;  
        dataStan['n_medSess_sensitivity']=2;
    elif model=='model7' and table=='tabel3' and group=='PD':
        dataStan['n_conds_alpha_pos']=1;    
        dataStan['n_medSess_alpha_pos']=1;  
        dataStan['n_conds_alpha_neg']=2;    
        dataStan['n_medSess_alpha_neg']=2;  
        dataStan['n_conds_weight']=2;       
        dataStan['n_medSess_weight']=2;     
        dataStan['n_conds_sensitivity']=2;  
        dataStan['n_medSess_sensitivity']=2;
    elif model=='model8' and table=='tabel3' and group=='PD':
        dataStan['n_conds_alpha_pos']=1;    
        dataStan['n_medSess_alpha_pos']=1;  
        dataStan['n_conds_alpha_neg']=2;    
        dataStan['n_medSess_alpha_neg']=1;  
        dataStan['n_conds_weight']=2;       
        dataStan['n_medSess_weight']=1;     
        dataStan['n_conds_sensitivity']=2;  
        dataStan['n_medSess_sensitivity']=1;
    
    ####### table 1
    # nConds array are fixed for all models
    elif table=='tabel1':  
        dataStan['n_conds_weight']=2;     
    return dataStan


def config_plot_model(model_calss:str, model_name:str, group:str):
    # configuration for both hierachical and individual plotting

    config_hier = {} 
    config_indv ={}
    #################### Tabel2
    if model_calss=='tabel2' and model_name=='model1':
        #hierarchical
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", "legend": ["Act", "Clr"], "range":(0,1)},
                       {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", "legend": ["Act", "Clr"], "range":(0,1)},
                       {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", "legend": ["Act", "Clr"], "range":(0,1)},
                       {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity", "legend": ["Act", "Clr"], "range":(0,.1)}]
        #individual
        config_indv = [{"param": "transfer_weight", "label": ["Weighting in Act", "Weighting in Clr"], "range":(0,1)},
                       {"param": "weight", "label": ["Weighting in Act", "Weighting in Clr"], "range":None},
                       {"param": "transfer_alpha_pos", "label": ["Positive learning rate in Act", "Positive learning rate in Clr"], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ["Negative learning rate in Act", "Negative learning rate in Clr"], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ["Sensitivity in Act", "Sensitivity in Clr"], "range":(0,.3)}]

    elif model_calss=='tabel2' and model_name=='model2':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", "legend": ["Act", "Clr"], "range":(0,1)},
                       {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", "legend": None, "range":(0,1)},
                       {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", "legend": None, "range":(0,1)},
                       {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity", "legend": ["Act", "Clr"], "range":(0,.1)}]
        config_indv = [{"param": "transfer_weight", "label": ["Weighting in Act", "Weighting in Clr"], "range":(0,1)},
                       {"param": "weight", "label": ["Weighting in Act", "Weighting in Clr"], "range":None},
                       {"param": "transfer_alpha_pos", "label": ["Positive learning rate"], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ["Negative learning rate"], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ["Sensitivity in Act", "Sensitivity in Clr"], "range":(0,.3)}]


    elif model_calss=='tabel2' and model_name=='model3':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", "legend": ["Act", "Clr"], "range":(0,1)},
                    {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", "legend": ["Act", "Clr"], "range":(0,1)},
                    {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", "legend": ["Act", "Clr"], "range":(0,1)},
                    {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity", "legend": None, "range":(0,.1)}]
        #individual
        config_indv = [{"param": "transfer_weight", "label": ["Weighting in Act", "Weighting in Clr"], "range":(0,1)},
                       {"param": "weight", "label": ["Weighting in Act", "Weighting in Clr"], "range":None},
                    {"param": "transfer_alpha_pos", "label": ["Positive learning rate in Act", "Positive learning rate in Clr"], "range":(0,1)},
                    {"param": "transfer_alpha_neg", "label": ["Negative learning rate in Act", "Negative learning rate in Clr"], "range":(0,1)},
                    {"param": "transfer_sensitivity", "label": ["Sensitivity"], "range":(0,.3)}]

    elif model_calss=='tabel2' and model_name=='model4':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", "legend": ["Act", "Clr"], "range":(0,1)},
                    {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", "legend": ["Act", "Clr"], "range":(0,1)},
                    {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", "legend": None, "range":(0,1)},
                    {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity", "legend": ["Act", "Clr"], "range":(0,.1)}]

        #individual
        config_indv = [{"param": "transfer_weight", "label": ["Weighting in Act", "Weighting in Clr"], "range":(0,1)},
                       {"param": "weight", "label": ["Weighting in Act", "Weighting in Clr"], "range":None},
                    {"param": "transfer_alpha_pos", "label": ["Positive learning rate in Act", "Positive learning rate in Clr"], "range":(0,1)},
                    {"param": "transfer_alpha_neg", "label": ["Negative learning rate in Act"], "range":(0,1)},
                    {"param": "transfer_sensitivity", "label": ["Sensitivity in Act", "Sensitivity in Clr"], "range":(0,.3)}]

    elif model_calss=='tabel2' and model_name=='model5':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", "legend": ["Act", "Clr"], "range":(0,1)},
                    {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", "legend": None, "range":(0,1)},
                    {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", "legend": ["Act", "Clr"], "range":(0,1)},
                    {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity", "legend": ["Act", "Clr"], "range":(0,.1)}]

        #individual
        config_indv = [{"param": "transfer_weight", "label": ["Weighting in Act", "Weighting in Clr"], "range":(0,1)},
                       {"param": "weight", "label": ["Weighting in Act", "Weighting in Clr"], "range":None},
                       {"param": "transfer_alpha_pos", "label": ["Positive learning rate"], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ["Negative learning rate in Act", "Negative learning rate in Clr"], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ["Sensitivity in Act", "Sensitivity in Clr"], "range":(0,.3)}]

    elif model_calss=='tabel2' and model_name=='model6':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", "legend": ["Act", "Clr"], "range":(0,1)},
                    {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", "legend": None, "range":(0,1)},
                    {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", "legend": None, "range":(0,1)},
                    {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity", "legend": None, "range":(0,.1)}]
            
        #individual
        config_indv = [{"param": "transfer_weight", "label": ["Weighting in Act", "Weighting in Clr"], "range":(0,1)},
                       {"param": "weight", "label": ["Weighting in Act", "Weighting in Clr"], "range":None},
                    {"param": "transfer_alpha_pos", "label": ["Positive learning rate"], "range":(0,1)},
                    {"param": "transfer_alpha_neg", "label": ["Negative learning rate",], "range":(0,1)},
                    {"param": "transfer_sensitivity", "label": ["Sensitivity"], "range":(0,.3)}]

    #################### Tabel3
    if model_calss=='tabel3' and model_name=='model1' and group=='HC':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", 
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", 
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", 
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity",
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,.1)}]
        #individual
        config_indv = [{"param": "transfer_weight", "label": ['Weighting in Act-Sess1', ' Weighting in Act-Sess2', 
                                                     'Weighting in Clr-Sess1', 'Weighting in Clr-Sess2'], "range":(0,1)},
                       {"param": "weight", "label": ['Weighting in Act-Sess1', ' Weighting in Act-Sess2', 
                                                     'Weighting in Clr-Sess1', 'Weighting in Clr-Sess2'], "range":None},                             
                       {"param": "transfer_alpha_pos", "label": ['Positive learning rate in Act-Sess1', 
                                                                 'Positive learning rate in Act-Sess2', 
                                                                 'Positive learning rate in Clr-Sess1', 
                                                                 'Positive learning rate in Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ['Negative learning rate in Act-Sess1', 
                                                                 'Negative learning rate in Act-Sess2', 
                                                                 'Negative learning rate in Clr-Sess1', 
                                                                 'Negative learning rate in Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ['Sensitivity in Act-Sess1', 
                                                                   'Sensitivity in Act-Sess2', 
                                                                   'Sensitivity in Clr-Sess1', 
                                                                   'Sensitivity in Clr-Sess2'], "range":(0,.3)}]

    elif model_calss=='tabel3' and model_name=='model2' and group=='HC':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", 
                        "legend": ['Act', 'Clr'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", 
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", 
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity",
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,.1)}]
        #individual
        config_indv = [{"param": "transfer_weight", "label": ['Weighting in Act', 'Weighting in Clr'], "range":(0,1)},
                       {"param": "weight", "label": ['Weighting in Act', 'Weighting in Clr'], "range":None},                             
                       {"param": "transfer_alpha_pos", "label": ['Positive learning rate in Act-Sess1', 
                                                                 'Positive learning rate in Act-Sess2', 
                                                                 'Positive learning rate in Clr-Sess1', 
                                                                 'Positive learning rate in Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ['Negative learning rate in Act-Sess1', 
                                                                 'Negative learning rate in Act-Sess2', 
                                                                 'Negative learning rate in Clr-Sess1', 
                                                                 'Negative learning rate in Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ['Sensitivity in Act-Sess1', 
                                                                   'Sensitivity in Act-Sess2', 
                                                                   'Sensitivity in Clr-Sess1', 
                                                                   'Sensitivity in Clr-Sess2'], "range":(0,.3)}]

    elif model_calss=='tabel3' and model_name=='model3' and group=='HC':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", 
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", 
                        "legend": ['Act', 'Clr'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", 
                        "legend": ['Act', 'Clr'], "range":(0,1)},
                       {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity",
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,.1)}]
        #individual
        config_indv = [{"param": "transfer_weight", "label": ['Weighting in Act-Sess1', ' Weighting in Act-Sess2', 
                                                     'Weighting in Clr-Sess1', 'Weighting in Clr-Sess2'], "range":(0,1)},
                       {"param": "weight", "label": ['Weighting in Act-Sess1', ' Weighting in Act-Sess2', 
                                                     'Weighting in Clr-Sess1', 'Weighting in Clr-Sess2'], "range":None},                             
                       {"param": "transfer_alpha_pos", "label": ['Positive learning rate in Act', 
                                                                 'Positive learning rate in Clr'], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ['Negative learning rate in Act', 
                                                                 'Negative learning rate in Clr'], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ['Sensitivity in Act-Sess1', 
                                                                   'Sensitivity in Act-Sess2', 
                                                                   'Sensitivity in Clr-Sess1', 
                                                                   'Sensitivity in Clr-Sess2'], "range":(0,.3)}]
        
    elif model_calss=='tabel3' and model_name=='model4' and group=='HC':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", 
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", 
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", 
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity",
                        "legend": ['Act', 'Clr'], "range":(0,.1)}]
        #individual
        config_indv = [{"param": "transfer_weight", "label": ['Weighting in Act-Sess1', ' Weighting in Act-Sess2', 
                                                     'Weighting in Clr-Sess1', 'Weighting in Clr-Sess2'], "range":(0,1)},
                       {"param": "weight", "label": ['Weighting in Act-Sess1', ' Weighting in Act-Sess2', 
                                                     'Weighting in Clr-Sess1', 'Weighting in Clr-Sess2'], "range":None},                             
                       {"param": "transfer_alpha_pos", "label": ['Positive learning rate in Act-Sess1', 
                                                                 'Positive learning rate in Act-Sess2', 
                                                                 'Positive learning rate in Clr-Sess1', 
                                                                 'Positive learning rate in Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ['Negative learning rate in Act-Sess1', 
                                                                 'Negative learning rate in Act-Sess2', 
                                                                 'Negative learning rate in Clr-Sess1', 
                                                                 'Negative learning rate in Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ['Sensitivity in Act',  
                                                                   'Sensitivity in Clr'], "range":(0,.3)}]

    elif model_calss=='tabel3' and model_name=='model5' and group=='HC':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", 
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", 
                        "legend": ['Act', 'Clr'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", 
                        "legend": ['Act', 'Clr'], "range":(0,1)},
                       {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity",
                        "legend": ['Act', 'Clr'], "range":(0,.1)}]
        #individual
        config_indv = [{"param": "transfer_weight", "label": ['Weighting in Act-Sess1', ' Weighting in Act-Sess2', 
                                                     'Weighting in Clr-Sess1', 'Weighting in Clr-Sess2'], "range":(0,1)},
                       {"param": "weight", "label": ['Weighting in Act-Sess1', ' Weighting in Act-Sess2', 
                                                     'Weighting in Clr-Sess1', 'Weighting in Clr-Sess2'], "range":None},                             
                       {"param": "transfer_alpha_pos", "label": ['Positive learning rate in Act', 
                                                                 'Positive learning rate in Clr'], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ['Positive learning rate in Act', 
                                                                 'Positive learning rate in Clr'], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ['Sensitivityin Act', 
                                                                  'Sensitivity in Clr'], "range":(0,.3)}]

    elif model_calss=='tabel3' and model_name=='model6' and group=='HC':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", 
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", 
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", 
                        "legend": ['Act', 'Clr'], "range":(0,1)},
                       {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity",
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,.1)}]
        #individual
        config_indv = [{"param": "transfer_weight", "label": ['Weighting in Act-Sess1', ' Weighting in Act-Sess2', 
                                                     'Weighting in Clr-Sess1', 'Weighting in Clr-Sess2'], "range":(0,1)},
                       {"param": "weight", "label": ['Weighting in Act-Sess1', ' Weighting in Act-Sess2', 
                                                     'Weighting in Clr-Sess1', 'Weighting in Clr-Sess2'], "range":None},                             
                       {"param": "transfer_alpha_pos", "label": ['Positive learning rate in Act-Sess1', 
                                                                 'Positive learning rate in Act-Sess2', 
                                                                 'Positive learning rate in Clr-Sess1', 
                                                                 'Positive learning rate in Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ['Negative learning rate in Act', 
                                                                 'Negative learning rate in Clr'], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ['Sensitivity in Act-Sess1', 
                                                                   'Sensitivity in Act-Sess2', 
                                                                   'Sensitivity in Clr-Sess1', 
                                                                   'Sensitivity in Clr-Sess2'], "range":(0,.3)}]

    elif model_calss=='tabel3' and model_name=='model7' and group=='HC':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", 
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", 
                        "legend": ['Act', 'Clr'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", 
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity",
                        "legend": ['Act-Sess1', 'Act-Sess2', 'Clr-Sess1', 'Clr-Sess2'], "range":(0,.1)}]
        #individual
        config_indv = [{"param": "transfer_weight", "label": ['Weighting in Act-Sess1', ' Weighting in Act-Sess2', 
                                                     'Weighting in Clr-Sess1', 'Weighting in Clr-Sess2'], "range":(0,1)},
                       {"param": "weight", "label": ['Weighting in Act-Sess1', ' Weighting in Act-Sess2', 
                                                     'Weighting in Clr-Sess1', 'Weighting in Clr-Sess2'], "range":None},                             
                       {"param": "transfer_alpha_pos", "label": ['Positive learning rate in Act',  
                                                                 'Positive learning rate in Clr'], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ['Negative learning rate in Act-Sess1', 
                                                                 'Negative learning rate in Act-Sess2', 
                                                                 'Negative learning rate in Clr-Sess1', 
                                                                 'Negative learning rate in Clr-Sess2'], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ['Sensitivity in Act-Sess1', 
                                                                   'Sensitivity in Act-Sess2', 
                                                                   'Sensitivity in Clr-Sess1', 
                                                                   'Sensitivity in Clr-Sess2'], "range":(0,.3)}]


    elif model_calss=='tabel3' and model_name=='model8' and group=='HC':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", 
                        "legend": ['Act', 'Clr'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", 
                        "legend": ['Act', 'Clr'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", 
                        "legend": ['Act', 'Clr'], "range":(0,1)},
                       {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity",
                        "legend": ['Act', 'Clr'], "range":(0,.1)}]
        #individual
        config_indv = [{"param": "transfer_weight", "label": ['Weighting in Act', 'Weighting in Clr'], "range":(0,1)},
                       {"param": "weight", "label": ['Weighting in Act', 'Weighting in Clr'], "range":None},                             
                       {"param": "transfer_alpha_pos", "label": ['Positive learning rate in Act',  
                                                                 'Positive learning rate in Clr'], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ['Negative learning rate in Act',  
                                                                 'Negative learning rate in Clr'], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ['Sensitivity in Act', 
                                                                   'Sensitivity in Clr'], "range":(0,.3)}]

    ################################## Table 3 in PD

    if model_calss=='tabel3' and model_name=='model1' and group=='PD':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", 
                        "legend": ['Act-OFF', 'Act-ON', 'Clr-OFF', 'Clr-ON'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", 
                        "legend": ['OFF', 'ON'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", 
                        "legend": ['Act-OFF', 'Act-ON', 'Clr-OFF', 'Clr-ON'], "range":(0,1)},
                       {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity",
                        "legend": ['Act-OFF', 'Act-ON', 'Clr-OFF', 'Clr-ON'], "range":(0,.1)}]
        #individual
        config_indv = [{"param": "transfer_weight", "label": ['Weighting in Act-OFF', ' Weighting in Act-ON', 
                                                     'Weighting in Clr-OFF', 'Weighting in Clr-ON'], "range":(0,1)},
                       {"param": "weight", "label": ['Weighting in Act-OFF', ' Weighting in Act-ON', 
                                                     'Weighting in Clr-OFF', 'Weighting in Clr-ON'], "range":None},                             
                       {"param": "transfer_alpha_pos", "label": ['Positive learning rate in OFF',  
                                                                 'Positive learning rate in ON'], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ['Negative learning rate in Act-OFF', 
                                                                 'Negative learning rate in Act-ON',
                                                                 'Negative learning rate in Clr-OFF', 
                                                                 'Negative learning rate in Clr-ON'], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ['Sensitivity in Act-OFF', 
                                                                   'Sensitivity in Act-ON', 
                                                                   'Sensitivity in Clr-OFF', 
                                                                   'Sensitivity in Clr-ON'], "range":(0,.3)}]

    elif model_calss=='tabel3' and model_name=='model2' and group=='PD':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", 
                        "legend": ['Act', 'Clr'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", 
                        "legend": ['OFF', 'ON'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", 
                        "legend": ['Act-OFF', 'Act-ON', 'Clr-OFF', 'Clr-ON'], "range":(0,1)},
                       {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity",
                        "legend": ['Act-OFF', 'Act-ON', 'Clr-OFF', 'Clr-ON'], "range":(0,.1)}]
        #individual
        config_indv = [{"param": "transfer_weight", "label": ['Weighting in Act', 'Weighting in Clr'], "range":(0,1)},
                       {"param": "weight", "label": ['Weighting in Act', 'Weighting in Clr'], "range":None},                             
                       {"param": "transfer_alpha_pos", "label": ['Positive learning rate in OFF', 
                                                                 'Positive learning rate in ON'], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ['Negative learning rate in Act-OFF', 
                                                                 'Negative learning rate in Act-ON',
                                                                 'Negative learning rate in Clr-OFF', 
                                                                 'Negative learning rate in Clr-ON'], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ['Sensitivity in Act-OFF', 
                                                                   'Sensitivity in Act-ON', 
                                                                   'Sensitivity in Clr-OFF', 
                                                                   'Sensitivity in Clr-ON'], "range":(0,.3)}]

    elif model_calss=='tabel3' and model_name=='model3' and group=='PD':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", 
                        "legend": ['Act-OFF', 'Act-ON', 'Clr-OFF', 'Clr-ON'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", 
                        "legend": None, "range":(0,1)},
                       {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", 
                        "legend": ['Act', 'Clr'], "range":(0,1)},
                       {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity",
                        "legend": ['Act-OFF', 'Act-ON', 'Clr-OFF', 'Clr-ON'], "range":(0,.1)}]
        #individual
        config_indv = [{"param": "transfer_weight", "label": ['Weighting in Act-OFF', ' Weighting in Act-ON', 
                                                     'Weighting in Clr-OFF', 'Weighting in Clr-ON'], "range":(0,1)},
                       {"param": "weight", "label": ['Weighting in Act-OFF', ' Weighting in Act-ON', 
                                                     'Weighting in Clr-OFF', 'Weighting in Clr-ON'], "range":None},                             
                       {"param": "transfer_alpha_pos", "label": ['Positive learning rate'], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ['Negative learning rate in Act', 
                                                                 'Negative learning rate in Clr'], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ['Sensitivity in Act-OFF', 
                                                                   'Sensitivity in Act-ON', 
                                                                   'Sensitivity in Clr-OFF', 
                                                                   'Sensitivity in Clr-ON'], "range":(0,.3)}]
        
    elif model_calss=='tabel3' and model_name=='model4' and group=='PD':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", 
                        "legend": ['Act-OFF', 'Act-ON', 'Clr-OFF', 'Clr-ON'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", 
                        "legend": ['OFF', 'ON'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", 
                        "legend": ['Act-OFF', 'Act-ON', 'Clr-OFF', 'Clr-ON'], "range":(0,1)},
                       {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity",
                        "legend": ['Act', 'Clr'], "range":(0,.1)}]
        #individual
        config_indv = [{"param": "transfer_weight", "label": ['Weighting in Act-OFF', ' Weighting in Act-ON', 
                                                     'Weighting in Clr-OFF', 'Weighting in Clr-ON'], "range":(0,1)},
                       {"param": "weight", "label": ['Weighting in Act-OFF', ' Weighting in Act-ON', 
                                                     'Weighting in Clr-OFF', 'Weighting in Clr-ON'], "range":None},                             
                       {"param": "transfer_alpha_pos", "label": ['Positive learning rate in OFF', 
                                                                 'Positive learning rate in ON'], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ['Negative learning rate in Act-OFF', 
                                                                 'Negative learning rate in Act-ON',
                                                                 'Negative learning rate in Clr-OFF', 
                                                                 'Negative learning rate in Clr-ON'], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ['Sensitivity in Act',  
                                                                   'Sensitivity in Clr'], "range":(0,.3)}]

    elif model_calss=='tabel3' and model_name=='model5' and group=='PD':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", 
                        "legend": ['Act-OFF', 'Act-ON', 'Clr-OFF', 'Clr-ON'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", 
                        "legend": None, "range":(0,1)},
                       {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", 
                        "legend": ['Act', 'Clr'], "range":(0,1)},
                       {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity",
                        "legend": ['Act', 'Clr'], "range":(0,.1)}]
        #individual
        config_indv = [{"param": "transfer_weight", "label": ['Weighting in Act-OFF', ' Weighting in Act-ON', 
                                                     'Weighting in Clr-OFF', 'Weighting in Clr-ON'], "range":(0,1)},
                       {"param": "weight", "label": ['Weighting in Act-OFF', ' Weighting in Act-ON', 
                                                     'Weighting in Clr-OFF', 'Weighting in Clr-ON'], "range":None},                             
                       {"param": "transfer_alpha_pos", "label": ['Positive learning rate'], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ['Negative learning rate in Act', 
                                                                 'Negative learning rate in Clr'], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ['Sensitivityin Act', 
                                                                  'Sensitivity in Clr'], "range":(0,.3)}]

    elif model_calss=='tabel3' and model_name=='model6' and group=='PD':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", 
                        "legend": ['Act-OFF', 'Act-ON', 'Clr-OFF', 'Clr-ON'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", 
                        "legend": ['OFF', 'ON'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", 
                        "legend": ['Act', 'Clr'], "range":(0,1)},
                       {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity",
                        "legend": ['Act-OFF', 'Act-ON', 'Clr-OFF', 'Clr-ON'], "range":(0,.1)}]
        #individual
        config_indv = [{"param": "transfer_weight", "label": ['Weighting in Act-OFF', ' Weighting in Act-ON', 
                                                     'Weighting in Clr-OFF', 'Weighting in Clr-ON'], "range":(0,1)},
                       {"param": "weight", "label": ['Weighting in Act-OFF', ' Weighting in Act-ON', 
                                                     'Weighting in Clr-OFF', 'Weighting in Clr-ON'], "range":None},                             
                       {"param": "transfer_alpha_pos", "label": ['Positive learning rate in OFF', 
                                                                 'Positive learning rate in ON'], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ['Negative learning rate in Act', 
                                                                 'Negative learning rate in Clr'], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ['Sensitivity in Act-OFF', 
                                                                   'Sensitivity in Act-ON', 
                                                                   'Sensitivity in Clr-OFF', 
                                                                   'Sensitivity in Clr-ON'], "range":(0,.3)}]

    elif model_calss=='tabel3' and model_name=='model7' and group=='PD':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", 
                        "legend": ['Act-OFF', 'Act-ON', 'Clr-OFF', 'Clr-ON'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", 
                        "legend": None, "range":(0,1)},
                       {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", 
                        "legend": ['Act-OFF', 'Act-ON', 'Clr-OFF', 'Clr-ON'], "range":(0,1)},
                       {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity",
                        "legend": ['Act-OFF', 'Act-ON', 'Clr-OFF', 'Clr-ON'], "range":(0,.1)}]
        #individual
        config_indv = [{"param": "transfer_weight", "label": ['Weighting in Act-OFF', ' Weighting in Act-ON', 
                                                     'Weighting in Clr-OFF', 'Weighting in Clr-ON'], "range":(0,1)},
                       {"param": "weight", "label": ['Weighting in Act-OFF', ' Weighting in Act-ON', 
                                                     'Weighting in Clr-OFF', 'Weighting in Clr-ON'], "range":None},                             
                       {"param": "transfer_alpha_pos", "label": ['Positive learning rate'], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ['Negative learning rate in Act-OFF', 
                                                                 'Negative learning rate in Act-ON',
                                                                 'Negative learning rate in Clr-OFF', 
                                                                 'Negative learning rate in Clr-ON'], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ['Sensitivity in Act-OFF', 
                                                                   'Sensitivity in Act-ON', 
                                                                   'Sensitivity in Clr-OFF', 
                                                                   'Sensitivity in Clr-ON'], "range":(0,.3)}]


    elif model_calss=='tabel3' and model_name=='model8' and group=='PD':
        config_hier = [{"param": "transfer_hier_weight_mu", "label": "Weighting", 
                        "legend": ['Act', 'Clr'], "range":(0,1)},
                       {"param": "transfer_hier_alpha_pos_mu", "label": "Positive learning rate", 
                        "legend": None, "range":(0,1)},
                       {"param": "transfer_hier_alpha_neg_mu", "label": "Negative learning rate", 
                        "legend": ['Act', 'Clr'], "range":(0,1)},
                       {"param": "transfer_hier_sensitivity_mu", "label": "Sensitivity",
                        "legend": ['Act', 'Clr'], "range":(0,.1)}]
        #individual
        config_indv = [{"param": "transfer_weight", "label": ['Weighting in Act', 'Weighting in Clr'], "range":(0,1)},
                       {"param": "weight", "label": ['Weighting in Act', 'Weighting in Clr'], "range":None},                             
                       {"param": "transfer_alpha_pos", "label": ['Positive learning rate'], "range":(0,1)},
                       {"param": "transfer_alpha_neg", "label": ['Negative learning rate in Act', 
                                                                 'Negative learning rate in Clr'], "range":(0,1)},
                       {"param": "transfer_sensitivity", "label": ['Sensitivity in Act', 
                                                                   'Sensitivity in Clr'], "range":(0,.3)}]

    return config_hier, config_indv


def initialStanActClr(readBehFile= PROJECT_NoNAN_BEH_ALL_FILE, group:str='PD',
                    alpha_pos_size=(2,2), alpha_neg_size=(2,2), sens_size=(2,2)):
    """
    Prepare initial samples for Stan for Action and Color conditions.
 
    Parameters
    ----------
    data : pd.DataFrame
        Behavioral data  
        Group to select: 'HC' for healthy controls or 'PD' for Parkinson's patients.

    Returns
    -------
    dataStan : dict
        Dictionary containing metadata.
    """
    # Load full dataset across all participants
    behAll = pd.read_csv(f"{readBehFile}")
 
    # Select only participants from the specified group 
    data = behAll[(behAll['patient'] == group)].copy().reset_index(drop=False)

    # Count number of participants 
    nParts = len(np.unique(data['sub_ID']))
 
    initials = []
    for _ in range(N_CHAIN):
        chaininit = {
            'hier_alpha_pos_mu'
            'z_alpha_pos': np.random.uniform(-1, 1, size=(nParts, *alpha_pos_size)),
            'z_alpha_neg': np.random.uniform(-1, 1, size=(nParts, *alpha_neg_size)),
            'z_sensitivity': np.random.uniform(-1, 1, size=(nParts, *sens_size)),
            'hier_alpha_sd': np.random.uniform(0.01, 0.1),
            'hier_sensitivity_sd': np.random.uniform(0.01, 0.02),
            'transfer_sensitivity': np.random.uniform(0.03, 0.07, size=(nParts, *sens_size))
        }
        initials.append(chaininit)


    return initials
  
def to_pickle(stan_fit, save_path):
    """Save pickle the fitted model's results with .pkl format.
    """
    try:
        with open(save_path, "wb") as f:   #Pickling
            pickle.dump({"fit" : stan_fit}, f, protocol=pickle.HIGHEST_PROTOCOL)       
            f.close()
            print('Saved results to ', save_path)
    except:
        print("An exception occurred")

def load_pickle(load_path):
    """Load model results from pickle.
    """
    try:
        with open(load_path, "rb") as fp:   # Unpickling
            results_load = pickle.load(fp)
            return results_load
    except:
        print("An exception occurred")
     
# Taken from https://github.com/laurafontanesi/rlssm/blob/main/rlssm/utils.py 
def waic_fun(log_likelihood):
    """Calculates the Watanabe-Akaike information criteria.
    Calculates pWAIC1 and pWAIC2
    according to http://www.stat.columbia.edu/~gelman/research/published/waic_understand3.pdf
    Parameters
    ----------
    pointwise : bool, default to False
        By default, gives the averaged waic.
        Set to True is you want additional waic per observation.
    Returns
    -------
    out: dict
        Dictionary containing lppd (log pointwise predictive density),
        p_waic, waic, waic_se (standard error of the waic), and
        pointwise_waic (when `pointwise` is True).
    """
    
    N = log_likelihood.shape[1]
    likelihood = np.exp(log_likelihood)

    mean_l = np.mean(likelihood, axis=0) # N observations

    pointwise_lppd = np.log(mean_l)
    lppd = np.sum(pointwise_lppd)

    pointwise_var_l = np.var(log_likelihood, axis=0) # N observations
    var_l = np.sum(pointwise_var_l)

    pointwise_waic = - 2*pointwise_lppd +  2*pointwise_var_l
    waic = -2*lppd + 2*var_l
    waic_se = np.sqrt(N * np.var(pointwise_waic))

    out = {'lppd':lppd,
           'p_waic':var_l,
           'waic':waic,
           'waic_se':waic_se}
    return out

def waic_models(model_calss:str, list_model:list[str]):
    # calcualte waic for all models in each group 
    for group in ['HC', 'PD']:     
        # declare waice variable
        waic_values = np.zeros(len(list_model))
        # loop over list of participants
        for i, model_name in enumerate(list_model):
            print(model_name)
            # The adrees name of pickle file
            pickelDir = f'{SCRATCH_HIER_MODEL_DIR}/{model_calss}/{model_name}/{group}/{model_calss}_{model_name}_{group}.pkl'
            print(pickelDir)
            #Loading the pickle file of model fit from the subject directory
            loadPkl = load_pickle(load_path=pickelDir)
            fit = loadPkl['fit'] 
            # get the linkelihood and comarision assessment       
            log_lik = fit['log_lik']
            criteria = waic_fun(log_likelihood=log_lik)
            waic_values[i] = criteria['waic']

        ## waic
        print(f'WAIC in {group} for: ',model_calss, ' : ', waic_values)
        #dwaic
        dWAIC = waic_values - np.min(waic_values)
        print(f'dWAIC in {group}  for: ',model_calss, ' : ',dWAIC)
        # realtive weight
        weight = [np.exp(-.5*dWAIC[i])/np.sum(np.exp(-.5*dWAIC)) for i in range(len(dWAIC))]
        print(f'weight in {group}  for: ',model_calss, ' : ', weight)


def MAP_last_axis(posterior_samples:np.ndarray):

    # Shape without the last dimension
    out_shape = posterior_samples.shape[:-1]
    map_estimates = np.zeros(out_shape)
    max_densities = np.zeros(out_shape)

    # Iterate over all indices except last axis
    for idx in np.ndindex(out_shape):
        samples_1d = posterior_samples[idx]  # shape: (last_dim,)

        # Evaluate KDE on a grid
        kde = gaussian_kde(samples_1d)
        x = np.linspace(samples_1d.min(), samples_1d.max(), 1000)
        density = kde(x)
        # MAP estimate = location of the maximum density
        max_idx = np.argmax(density)
        map_estimates[idx] = x[max_idx]
        max_densities[idx] = density[max_idx]

    return map_estimates, max_densities


def participant_list(readBehFile= PROJECT_NoNAN_BEH_ALL_FILE, group:str='PD'):
    """
    return participant list for each gorup

    Parameters
    ----------
    data : pd.DataFrame
        Behavioral data  
        Group to select: 'HC' for healthy controls or 'PD' for Parkinson's patients.

    Returns
    -------
    dataStan : dict
        Dictionary containing standardized data arrays 
    """
    # Load full dataset across all participants
    behAll = pd.read_csv(f"{readBehFile}")

    # Select only participants from the specified group 
    data = behAll[(behAll['patient'] == group)].copy().reset_index(drop=False)

    # participant list
    participants = data['sub_ID'].unique()

    return participants

def save_indv_summary_posterior(fit: dict[str, np.ndarray], model_dir: str,param: str,group: str,model_name: str):
    """
    Save mean posterior for individual parameters:
    Expected shape: (nParts, nConds, nSess, nSamples)
    """
    
    param_post = fit[param]
    print('param_post.shape:', param_post.shape)

    # Ensure 4D shape
    if param_post.ndim == 4:
        pass
    elif param_post.ndim == 3:
        param_post = param_post[:, :, np.newaxis, :]
    elif param_post.ndim == 2:
        param_post = param_post[:, np.newaxis, np.newaxis, :]
    else:
        raise ValueError("Unsupported parameter shape")

    nParts, nConds, nSess, nSamples = param_post.shape

    if nConds==2:
        conditions = ['Act','Stim']
    else:
        conditions = ['ActStim']

    # Get participant names
    participants_names = participant_list(readBehFile=PROJECT_NoNAN_BEH_ALL_FILE,group=group)

    # Safety check (this can silently break otherwise)
    if len(participants_names) != nParts:
        raise ValueError(
            f"Mismatch: {len(participants_names)=} vs {nParts=}"
        )
    # Mean over samples
    if group=='PD':
        # list of medications
        if nSess==2:
            medciations = ['OFF','ON']
        else:
            medciations = ['OFFON']

        # map and mean of posterior
        param_post_map, _ = MAP_last_axis(param_post)
        param_post_mean = param_post.mean(axis=-1)
        # Collect rows
        rows = []
        for p_idx in range(nParts):
            participant = participants_names[p_idx]

            for c_idx, condition in enumerate(conditions):
                for s_idx, medciation in enumerate(medciations):
                    # reverse the value for stimulus weighting parameter
                    if condition == 'Stim':
                        param_post_map_magnitude = -1*param_post_map[p_idx, c_idx, s_idx]
                    else:
                        param_post_map_magnitude = param_post_map[p_idx, c_idx, s_idx]
                    
                    rows.append({
                        'patient':group,
                        'medication': medciation,
                        'sub_ID': participant,
                        'block': condition,
                        f'{param}_parameter_mean': param_post_mean[p_idx, c_idx, s_idx],
                        f'{param}_parameter_map': param_post_map[p_idx, c_idx, s_idx],
                        f'{param}_parameter_map_magnitude': param_post_map_magnitude
                    })
    if group=='HC': 
        #map and mean of posterior, average accross sessions
        param_post_mean = param_post.mean(axis=-1).mean(axis=-1)
        param_post_map,_ = MAP_last_axis(param_post)
        param_post_map = param_post_map.mean(axis=-1)
        
        # Collect rows
        rows = []
        for p_idx in range(nParts):
            participant = participants_names[p_idx]

            for c_idx, condition in enumerate(['Act','Stim']):
                # reverse the value for stimulus weighting parameter
                if condition == 'Stim':
                    param_post_map_magnitude = -1*param_post_map[p_idx, c_idx]
                else:
                    param_post_map_magnitude = param_post_map[p_idx, c_idx]
                    
                rows.append({
                    'patient':group,
                    'medication': 'OFF',
                    'sub_ID': participant,
                    'block': condition,
                    f'{param}_parameter_mean': param_post_mean[p_idx, c_idx],
                    f'{param}_parameter_map': param_post_map[p_idx, c_idx],
                    f'{param}_parameter_map_magnitude': param_post_map_magnitude
                })

    # Create DataFrame once
    df = pd.DataFrame(rows)

    # Save
    df.to_csv(f'{model_dir}/{model_name}_{group}_{param}.csv', index=False)
    print(f'{model_dir}/{model_name}_{group}_{param}.csv')

# merge and save behavioral data proportions and summary measurements of model parameters
def combine_parameter_highRewardChoice(main_indv_model_dir: str, param: str, model: str):

    # Load parameter data
    df_parameter_PD = pd.read_csv(f'{main_indv_model_dir}/PD/{model}_PD_{param}.csv')
    df_parameter_HC = pd.read_csv(f'{main_indv_model_dir}/HC/{model}_HC_{param}.csv')
 
    # Combine
    df_parameter = pd.concat([df_parameter_PD, df_parameter_HC], ignore_index=True)

    # Load behavioral data
    behALL_high_reward_groupby = pd.read_csv(PROJECT_NoNAN_BEH_REL_IRREL_HIGH_REWARD_OPTION_GROUPBY_ALL_FILE)

    # Merge
    df_merged = pd.merge(
        behALL_high_reward_groupby,
        df_parameter,
        on=['patient', 'medication', 'sub_ID', 'block'],
        how='inner'
    )

    # save behavioral data fwith relevant and irrelevant high reward options and summary of model parameter weights
    df_merged.to_csv(f"{PROJECT_NoNAN_BEH_REL_IRREL_HIGH_REWARD_OPTION_GROUPBY_ALL_FILE_MODEL_PARAMETER}",index=False)
 
# invery logit
def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

def log1p_exp(p):
    return np.log(1 + np.exp(p))

# add grand truth parameters to behavioral raw data
def generating_hier_grand_truth(hier_weight_mu, hier_alphaAct_pos_mu, hier_alphaAct_neg_mu,
                                hier_alphaClr_pos_mu, hier_alphaClr_neg_mu, hier_sensitivity_mu,
                                hier_alpha_sd, hier_weight_sd,  hier_sensitivity_sd,
                                sim):
    #generate data and put individual and heirarchical true parameters into task desgin for each participant 
    try:
        # read collected data across data
        rawBehAll = pd.read_csv(PROJECT_RAW_BEH_ALL_FILE)
        # removing some participnats due to lack of one session
        withdraw_subs = ['sub-057', 'sub-076', 'sub-091']
        for sub in withdraw_subs:
            rawBehAll = rawBehAll[rawBehAll['sub_ID']!=sub].reset_index(drop=False)
            
            # list of subjects
        subList = rawBehAll['sub_ID'].unique()
        # Get the partisipant's task design from the original behavioral dataset 'originalfMRIbehFiles'
        rawBehAll = rawBehAll.rename(columns={'leftCanBePushed                ': 'leftCanBePushed'})

        # choose some relevant columns
        task_design_parameter = rawBehAll[['session', 'run', 'stimActFirst', 'block', 'stimActBlock', 'trialNumber', 'yellowOnLeftSide', 'leftCanBePushed', 'winAmtLeft', 'winAmtRight', 'winAmtYellow', 'winAmtBlue', 'winAmtPushable', 'winAmtPullable', 'yellowCorrect', 'pushCorrect', 'reverse', 'group', 'patient']].copy()
        # Put true parameters into the task design, define new columns of grand truth parameters within predefined task design 
        task_design_parameter[['transfer_alphaAct_pos', 'transfer_alphaAct_neg', 'transfer_alphaClr_pos', 'transfer_alphaClr_neg', 'transfer_weight', 'transfer_sensitivity']] = ""  
       
        # Set true parameters for each session and conditions realted to unkown parameters
        for subName in subList:
            # extract subject data
            rawBehAll_subj = rawBehAll[rawBehAll['sub_ID']==subName].reset_index(drop=False)
            group_subj = rawBehAll_subj['patinet'].uniqie()
            # group index, HC 0, PD 1
            patient_index = 0
            if group_subj=='HC':
                patient_index = 0
            elif group_subj=='PD':
                patient_index = 1
            # generate new samples from hierarhcial parameters
            transfer_weight = inv_logit(np.random.normal(hier_weight_mu[patient_index],hier_weight_sd[patient_index]))
            transfer_alphaAct_pos = inv_logit(np.random.normal(hier_alphaAct_pos_mu[patient_index],hier_alpha_sd[patient_index]))
            transfer_alphaAct_neg = inv_logit(np.random.normal(hier_alphaAct_neg_mu[patient_index],hier_alpha_sd[patient_index]))
            transfer_alphaClr_pos = inv_logit(np.random.normal(hier_alphaClr_pos_mu[patient_index],hier_alpha_sd[patient_index]))
            transfer_alphaClr_neg = inv_logit(np.random.normal(hier_alphaClr_neg_mu[patient_index],hier_alpha_sd[patient_index]))
            transfer_sensitivity = log1p_exp(np.random.normal(hier_sensitivity_mu[patient_index],hier_sensitivity_sd[patient_index]))

            # Put generated true parameters within the predefined task design dataframe
            for condition, block in enumerate(['Act', 'Stim']):
                filter =  (task_design_parameter['block'] == block) & (rawBehAll['sub_ID']==subName)
                task_design_parameter.loc[filter, 'transfer_alphaAct_pos'] = transfer_alphaAct_pos[patient_index]
                task_design_parameter.loc[filter, 'transfer_alphaAct_neg'] = transfer_alphaAct_neg[patient_index]
                task_design_parameter.loc[filter, 'transfer_alphaClr_pos'] = transfer_alphaClr_pos[patient_index]
                task_design_parameter.loc[filter, 'transfer_alphaClr_neg'] = transfer_alphaClr_neg[patient_index]
                task_design_parameter.loc[filter, 'transfer_weight'] = transfer_weight[patient_index, condition]
                task_design_parameter.loc[filter, 'transfer_sensitivity'] = transfer_sensitivity[patient_index]
       
         # Check existing directory of subject name forlder and simulation number
        if not os.path.isdir(f'{SCRATCH_RAW_BEH_ALL_PARAMETER_GRAND_TRUTTH_DIR}/{str(sim)}'):
            os.makedirs(f'{SCRATCH_RAW_BEH_ALL_PARAMETER_GRAND_TRUTTH_DIR}/{str(sim)}') 

        # Save task design plus true parameters for each participant
        task_design_parameter.to_csv(f'{SCRATCH_RAW_BEH_ALL_PARAMETER_GRAND_TRUTTH_DIR}/{str(sim)}/task_design_parameter_{str(sim)}.csv', index=False)
    
        # datafram of hierarchical true parameters
        dictionary =  {'hier_alphaAct_pos_mu':hier_alphaAct_pos_mu,
                      'hier_alphaAct_neg_mu': hier_alphaAct_neg_mu,
                      'hier_alphaClr_pos_mu': hier_alphaClr_pos_mu,
                      'hier_alphaClr_neg_mu': hier_alphaClr_neg_mu,
                      'hier_weight_mu': hier_weight_mu,
                      'hier_sensitivity_mu': hier_sensitivity_mu,
                      'hier_alpha_sd': hier_alpha_sd,
                      'hier_weight_sd': hier_weight_sd,
                      'hier_sensitivity_sd': hier_sensitivity_sd}
        # Writing to sample.json
        with open(f'{SCRATCH_RAW_BEH_ALL_PARAMETER_GRAND_TRUTTH_DIR}/{str(sim)}/hier_parameter_grandtruth_{str(sim)}.json', 'w') as outfile:
            json.dump(dictionary, outfile)

        return print("All true parameters for each participant have been generated and saved successfully!")
    except Exception as e:
        return print("An exception accured within generating_hier_grand_truth function: " + str(e))

# Simulated data for each participatn based on predefined True Parameters
def simulate_hier_rl(sim):
    # Read predefined task design with true parameters
    task_design_parameter = pd.read_csv(f'{SCRATCH_RAW_BEH_ALL_PARAMETER_GRAND_TRUTTH_DIR}/{str(sim)}/task_design_parameter_{str(sim)}.csv')
    # list of subjects
    subList = task_design_parameter['sub_ID'].unique()
    # saved simulated data
    simulation_data_parameter = task_design_parameter.copy()
    # loop across participants
    for subName in subList:
     #Simulated data from the predefined true parameters in dataframe simulation_data_parameter_param"""
        for session in [1, 2]: # session
            for run in [1, 2]: # two distinct environemnt
                for condition in ['Act', 'Stim']: # condition
                    # booloian filter
                    filter = (simulation_data_parameter['block']==condition)&(simulation_data_parameter['session']==session)\
                        &(simulation_data_parameter['run']==run)&(simulation_data_parameter['sub_ID']==subName)
                    # get some relevant part data
                    simulation_data_parameter_param_split = simulation_data_parameter[filter]  

                    # Predefined conditions for each trial
                    block = simulation_data_parameter_param_split.block.to_numpy()
                    
                    # Predefined Winning amout of reward for Action and Color options
                    winAmtPushable = simulation_data_parameter_param_split['winAmtPushable'].to_numpy()
                    winAmtPullable = simulation_data_parameter_param_split['winAmtPullable'].to_numpy()
                    winAmtYellow = simulation_data_parameter_param_split['winAmtYellow'].to_numpy()
                    winAmtBlue = simulation_data_parameter_param_split['winAmtBlue'].to_numpy()  
                    
                    # Predefined options on left and right side
                    leftCanBePushed = simulation_data_parameter_param_split['leftCanBePushed'].to_numpy()
                    yellowOnLeftSide = simulation_data_parameter_param_split['yellowOnLeftSide'].to_numpy()
                    
                    # Predefined Correct responces for Action and color options
                    pushCorrect = simulation_data_parameter_param_split['pushCorrect'].to_numpy()
                    yellowCorrect = simulation_data_parameter_param_split['yellowCorrect'].to_numpy()
                    
                    # Predefined Ground truth Parameters
                    transfer_alphaAct_pos = simulation_data_parameter_param_split['transfer_alphaAct_pos'].to_numpy()
                    transfer_alphaAct_neg = simulation_data_parameter_param_split['transfer_alphaAct_neg'].to_numpy()
                    transfer_alphaClr_pos = simulation_data_parameter_param_split['transfer_alphaClr_pos'].to_numpy()
                    transfer_alphaClr_neg = simulation_data_parameter_param_split['transfer_alphaClr_neg'].to_numpy()
                    transfer_weight = simulation_data_parameter_param_split['transfer_weight'].to_numpy()
                    transfer_sensitivity = simulation_data_parameter_param_split['transfer_sensitivity'].to_numpy()

                    
                    # Predefined Number of trials
                    n_trials = simulation_data_parameter_param_split.shape[0]
    
                    # Output of simulation for correct choice and Action and Color chosen
                    correctChoice = np.zeros(n_trials).astype(int)
                    pushed = np.zeros(n_trials).astype(int)
                    yellowChosen = np.zeros(n_trials).astype(int)

                    # Initial reward probability
                    p_push = .5
                    p_yell = .5
                        
                    # Loop over trials
                    for i in range(n_trials):
                        
                        # Compute the Standard Expected Value of each seperated option 
                        EV_push = p_push*winAmtPushable[i]
                        EV_pull = (1-p_push)*winAmtPullable[i]
                        EV_yell = p_yell*winAmtYellow[i]
                        EV_blue = (1-p_yell)*winAmtBlue[i]

                        # Relative contribution of Action Value LeexpValuePusharning verus Color Value Learning by combining the expected values of option
                        EV_push_yell = transfer_weight[i]*EV_push + (1 - transfer_weight[i])*EV_yell;
                        EV_push_blue = transfer_weight[i]*EV_push + (1 - transfer_weight[i])*EV_blue;
                        EV_pull_yell = transfer_weight[i]*EV_pull + (1 - transfer_weight[i])*EV_yell;
                        EV_pull_blue = transfer_weight[i]*EV_pull + (1 - transfer_weight[i])*EV_blue;

                        # Calculating the soft-max function based on (pushed and yellow) vs (pulled and blue) 
                        if (leftCanBePushed[i] == 1 and yellowOnLeftSide[i] == 1) or (leftCanBePushed[i] == 0 and yellowOnLeftSide[i] == 0):
                            # Applying soft-max function 
                            nom = np.exp(transfer_sensitivity[i]*EV_push_yell)
                            denom = nom + np.exp(transfer_sensitivity[i]*EV_pull_blue)
                            theta = nom/denom
                            # Make a binary choice response by bernouli 
                            y = np.random.binomial(1, p=theta, size=1) 
                            # Calculating to which Action vs Color Response response
                            if y==1:
                                pushed[i] = 1
                                yellowChosen[i] = 1
                            else:
                                pushed[i] = 0
                                yellowChosen[i] = 0
                        # Calculating the soft-max function based on (pushed and blue) vs (pulled and yellow) 
                        elif (leftCanBePushed[i] == 1 and yellowOnLeftSide[i] == 0) or (leftCanBePushed[i] == 0 and yellowOnLeftSide[i] == 1):
                            # Applying soft-max function 
                            nom = np.exp(transfer_sensitivity[i]*EV_push_blue)
                            denom = nom + np.exp(transfer_sensitivity[i]*EV_pull_yell)
                            theta = nom/denom
                            # Make a binary choice response by bernouli 
                            y = np.random.binomial(1, p=theta, size=1)
                            # Make a choice based on the probability 
                            if y==1:
                                pushed[i] = 1
                                yellowChosen[i] = 0
                            else:
                                pushed[i] = 0
                                yellowChosen[i] = 1

                        if block[i] == 'Act':
                            # Get reward based on the simulated response
                            correctChoice[i] = int(pushed[i] == pushCorrect[i])
                            
                            # Rl rule update over Action Learning Values for the next trial
                            if pushed[i] == 1:
                                if correctChoice[i]:
                                    p_push = p_push + transfer_alphaAct_pos[i]*(correctChoice[i] - p_push)
                                else:
                                    p_push = p_push + transfer_alphaAct_neg[i]*(correctChoice[i] - p_push)
                            elif pushed[i] == 0:
                                if correctChoice[i]>0:
                                    p_push = p_push - transfer_alphaAct_pos[i]*(correctChoice[i] + p_push -1)
                                else:
                                    p_push = p_push - transfer_alphaAct_neg[i]*(correctChoice[i] + p_push -1)
                                    
                        elif block[i] == 'Stim':
                            correctChoice[i] = int(yellowChosen[i] == yellowCorrect[i])

                                # Rl rule update Color Action Learning values for the next trial
                            if yellowChosen[i] == 1:
                                if correctChoice[i]>0:
                                    p_yell = p_yell + transfer_alphaClr_pos[i]*(correctChoice[i] - p_yell)
                                else:
                                    p_yell = p_yell + transfer_alphaClr_neg[i]*(correctChoice[i] - p_yell)
                            elif yellowChosen[i] == 0:
                                if correctChoice[i]>0:
                                    p_yell = p_yell - transfer_alphaClr_pos[i]*(correctChoice[i] + p_yell -1)
                                else:
                                    p_yell = p_yell - transfer_alphaClr_neg[i]*(correctChoice[i] + p_yell -1)
                    

                    # output results
                    simulation_data_parameter.loc[filter, 'pushed'] = pushed  
                    simulation_data_parameter.loc[filter, 'yellowChosen'] = yellowChosen  
                    simulation_data_parameter.loc[filter, 'correctChoice'] = correctChoice  
    


    simulation_data_parameter.to_csv(f'{SCRATCH_RAW_BEH_ALL_PARAMETER_GRAND_TRUTTH_DIR}/{str(sim)}/simulation_data_parameter{str(sim)}.csv', index=False)
    
    return print("All simulations have been done successfully!")
     
