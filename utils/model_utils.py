import numpy as np
import pickle
import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from utils import config

def compute_and_save_clinical_parameters(
    readClicalEvalFile=config.PROJECT_CLIN_EVAL_FILE,
    readModel=config.SCRATCH_HIER_MODEL_DIR,
    outDir=config.SCRATCH_CLIN_EVAL_DIR
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

    # -------------------------------------------------------
    # Helper function: KDE mode
    # -------------------------------------------------------
    def get_mode_density(values):
        """Return the mode of a posterior distribution using KDE."""
        kde = gaussian_kde(values)
        x_grid = np.linspace(min(values), max(values), 1000)
        return x_grid[np.argmax(kde(x_grid))]

    # -------------------------------------------------------
    # Load clinical evaluation
    # -------------------------------------------------------
    clinical_evaluation = pd.read_csv(f'{readClicalEvalFile}')

    # =======================================================
    # --------------  LOAD PD MODEL RESULTS  -----------------
    # =======================================================
    pkl_PD = f'{readModel}/Hier-RL-Model/Tabel3/PD/tabel3_model1_complement_prob_PD.pkl'
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

    # =======================================================
    # --------------  LOAD HC MODEL RESULTS  -----------------
    # =======================================================
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

    # =======================================================
    # -------- MERGE MODEL PARAMETERS WITH CLINICAL DATA ----
    # =======================================================
    parameter_clinical_evaluation = clinical_evaluation.copy()

    mask_HC = parameter_clinical_evaluation['group'] == 'HC'
    mask_PD = parameter_clinical_evaluation['group'] == 'PD'

    # Assign parameters
    parameter_clinical_evaluation.loc[mask_HC, 'map_mean_alpha_pos'] = map_mean_alpha_pos_HC
    parameter_clinical_evaluation.loc[mask_PD, 'map_mean_alpha_pos'] = map_mean_alpha_pos_PD

    parameter_clinical_evaluation.loc[mask_HC, 'map_mean_alpha_neg'] = map_mean_alpha_neg_HC
    parameter_clinical_evaluation.loc[mask_PD, 'map_mean_alpha_neg'] = map_mean_alpha_neg_PD

    parameter_clinical_evaluation.loc[mask_HC, 'map_mean_sensitivity'] = map_mean_sensitivity_HC
    parameter_clinical_evaluation.loc[mask_PD, 'map_mean_sensitivity'] = map_mean_sensitivity_PD

    parameter_clinical_evaluation.loc[mask_HC, 'map_mean_weighting_act'] = map_mean_weighting_act_HC
    parameter_clinical_evaluation.loc[mask_PD, 'map_mean_weighting_act'] = map_mean_weighting_act_PD

    parameter_clinical_evaluation.loc[mask_HC, 'map_mean_weighting_clr'] = map_mean_weighting_clr_HC
    parameter_clinical_evaluation.loc[mask_PD, 'map_mean_weighting_clr'] = map_mean_weighting_clr_PD

    parameter_clinical_evaluation.loc[mask_HC, 'map_mean_weighting'] = map_mean_weighting_HC
    parameter_clinical_evaluation.loc[mask_PD, 'map_mean_weighting'] = map_mean_weighting_PD

    # PD-specific medication effects
    parameter_clinical_evaluation.loc[mask_PD, 'map_med_alpha_pos'] = map_med_alpha_pos_PD
    parameter_clinical_evaluation.loc[mask_PD, 'map_med_alpha_neg'] = map_med_alpha_neg_PD
    parameter_clinical_evaluation.loc[mask_PD, 'map_med_sensitivity'] = map_med_sensitivity_PD
    parameter_clinical_evaluation.loc[mask_PD, 'map_med_weighting_act'] = map_med_weighting_act_PD
    parameter_clinical_evaluation.loc[mask_PD, 'map_med_weighting_clr'] = map_med_weighting_clr_PD
    parameter_clinical_evaluation.loc[mask_PD, 'map_med_weighting'] = map_med_weighting_PD

    # UPDRS difference
    parameter_clinical_evaluation.loc[mask_PD, 'med_UPDRS'] = \
        parameter_clinical_evaluation['total_UPDRSON'] - parameter_clinical_evaluation['total_UPDRSOFF']

    # -------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------
 
    # Check out if it does not exist
    if not os.path.isdir(f'{outDir}'):
            os.makedirs(f'{outDir}') 

    parameter_clinical_evaluation.to_csv(outDir, index=False)

    print(f"Saved clinical parameter table to:\n{outDir}")


def dataStanActClr(readBehFile= config.PROJECT_NoNAN_BEH_ALL_FILE, group:str='PD',
                    lr_pos_size=(2,2), lr_neg_size=(2,2), lr_sens_size=(2,2)):
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
        Dictionary containing standardized data arrays and metadata.
    """
    # Load full dataset across all participants
    behAll = pd.read_csv(f"{readBehFile}")

    # ------------------- Select only participants from the specified group -------------------
    data = behAll[(behAll['patient'] == group)].copy().reset_index(drop=False)

    # ------------------- Count number of participants -------------------
    nParts = len(np.unique(data['sub_ID']))

    # ------------------- Convert participant IDs to consecutive integer indices -------------------
    # Example: if sub_IDs are [101, 203, 405], they will become [1, 2, 3]
    data['sub_ID'] = data['sub_ID'].replace(np.unique(data.sub_ID), np.arange(1, nParts + 1)).astype(int)

    # ------------------- Number of conditions -------------------
    nConds = 2  # 1 = Action (Act), 2 = Color (Stim)

    # ------------------- Convert condition labels to integers -------------------
    # 'Act' -> 1, 'Stim' -> 2
    data['block'] = data['block'].replace(['Act', 'Stim'], [1, 2]).astype(int)

    # ------------------- Number of sessions / medication conditions -------------------
    nMeds_nSes = 2

    # ------------------- Set session or medication variable based on group -------------------
    if group == 'HC':
        # For healthy controls, use session column directly
        medication_session = np.array(data['session']).astype(int)
    elif group == 'PD':
        # For PD patients, map group labels: 1 -> OFF, 3 -> ON
        data['medication'] = data['group'].replace([1, 3], [1, 2]).astype(int)
        medication_session = np.array(data['medication']).astype(int)

    # ------------------- Organize data into a dictionary -------------------
    # Each key will be used for modeling or analysis (e.g., in Stan or other frameworks)
    dataStan = {
        'N': data.shape[0],  # Total number of trials
        'nParts': nParts,    # Number of participants
        'pushed': np.array(data['pushed']).astype(int),           # Action choice (push=1, pull=0)
        'yellowChosen': np.array(data['yellowChosen']).astype(int),  # Color choice (yellow=1, blue=0)
        'winAmtPushable': np.array(data['winAmtPushable']),       # Reward if pushed
        'winAmtPullable': np.array(data['winAmtPullable']),       # Reward if pulled
        'winAmtYellow': np.array(data['winAmtYellow']),           # Reward if yellow chosen
        'winAmtBlue': np.array(data['winAmtBlue']),               # Reward if blue chosen
        'rewarded': np.array(data['correctChoice']).astype(int),  # Whether choice was correct
        'participant': np.array(data['sub_ID']).astype(int),      # Participant index
        'indicator': np.array(data['indicator']).astype(int),     # Trial indicator variable
        'nConds': nConds,                                        # Number of conditions
        'condition': np.array(data['block']).astype(int),         # Condition per trial (1=Act, 2=Clr)
        'nMeds_nSes': nMeds_nSes,                                # Number of sessions or medication conditions
        'medication_session': medication_session                  # Session or medication index
    }

    initials = []
    for _ in range(config.N_CHAIN):
        chaininit = {
            'z_alpha_pos': np.random.uniform(-1, 1, size=(nParts, *lr_pos_size)),
            'z_alpha_neg': np.random.uniform(-1, 1, size=(nParts, *lr_neg_size)),
            'z_sensitivity': np.random.uniform(-1, 1, size=(nParts, *lr_sens_size)),
            'hier_alpha_sd': np.random.uniform(0.01, 0.1),
            'hier_sensitivity_sd': np.random.uniform(0.01, 0.02),
            'transfer_sensitivity': np.random.uniform(0.03, 0.07, size=(nParts, lr_sens_size))
        }
        initials.append(chaininit)


    return dataStan, initials


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
def waic(log_likelihood):
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
