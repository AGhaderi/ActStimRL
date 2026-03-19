# Base directory for Bheavioral data
PROJECT_DATA_DIR = "/mnt/projects/7TPD/bids/derivatives/fMRI_DA"
PROJECT_BEH_ALL_DATA_DIR = "/mnt/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData"
FIGURES_DIR = "/home/amingk/Documents/7TPD/ActStimRL/figures"
OUTPUT_CSV_DIR = "/home/amingk/Documents/7TPD/TransformedData"
PROJECT_REDCAP_DIR='/mnt/projects/7TPD/Documents/redcap'
PROJECT_CLIN_EVAL_DIR = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/Clinical_evaluation'
PROJECT_HIER_MODEL_DIR = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/Hier-RL-Model'
STAN_DIR = '/mrhome/amingk/Documents/7TPD/ActStimRL/stan_models'

####################### write in scratch folder
SCRATCH_BEH_ALL_DATA_DIR = "/mnt/scratch/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData"
SCRATCH_CLIN_EVAL_DIR = '/mnt/scratch/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/Clinical_evaluation'
SCRATCH_HIER_MODEL_DIR = '/mnt/scratch/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/Hier-RL-Model'

###################### color and opacity for HC, PD-OFF, PD-ON
COLORS = {'HC': "#5ea0ed", 'HC-Sess1': "#83adf1", 'HC-Sess2': '#0171be', 'PD-OFF': '#ff7b7b', 'PD-ON':"#cc0000", 
          'HC-POS':'#2dadf2', 'HC-NEG':'#a8d8f5', 'PD-POS':'#ee6969', 'PD-NEG':'#f4c1c1'}
OPACITY = {'HC': 1, 'PD-OFF':.5, 'PD-ON':.9}

####### csv file directories
#  raw behavioral data for both read and write
PROJECT_RAW_BEH_ALL_FILE = f"{PROJECT_BEH_ALL_DATA_DIR}/rawBehAll.csv" 
# NoNAN behavioral data for both read and write
PROJECT_NoNAN_BEH_ALL_FILE = f"{PROJECT_BEH_ALL_DATA_DIR}/NoNanBehAll.csv"
# clinical evalueion file
PROJECT_CLIN_EVAL_FILE = f'{PROJECT_CLIN_EVAL_DIR}/clinical_eval.csv'
# clinical evalueion file
PROJECT_MAP_CLIN_EVAL_FILE = f'{PROJECT_CLIN_EVAL_DIR}/map_clinical_eval.csv'
# behavioral data for both read and write, inclduing relevant and irrelevant high reward options
PROJECT_NoNAN_BEH_REL_IRREL_HIGH_REWARD_OPTION_ALL_FILE = f"{PROJECT_BEH_ALL_DATA_DIR}/NoNanBehAll_RelIrrelHighReward.csv"
# behavioral data for both read and write, inclduing relevant and irrelevant high reward options, avegrage across phases
PROJECT_NoNAN_BEH_REL_IRREL_HIGH_REWARD_OPTION_GROUPBY_ALL_FILE = f"{PROJECT_BEH_ALL_DATA_DIR}/NoNanBehAll_RelIrrelHighReward_Groupby.csv"

########### model fit setting
# Number of chains in MCMC procedure
N_CHAIN = 8
# The number of iteration or samples for each chain in MCM procedure
N_SAMPLES=3000
# number of warp up samples
N_WARMUP = 1000