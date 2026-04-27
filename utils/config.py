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
SCRATCH_INDV_MODEL_DIR = '/mnt/scratch/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/Indv-RL-Model'

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
SCARTCH_CLIN_EVAL_FILE = f'{SCRATCH_CLIN_EVAL_DIR}/clinical_eval.csv'
# clinical evalueion file
PROJECT_MAP_CLIN_EVAL_FILE = f'{PROJECT_CLIN_EVAL_DIR}/map_clinical_eval.csv'
# behavioral data for both read and write, inclduing relevant and irrelevant high reward options
PROJECT_NoNAN_BEH_REL_IRREL_HIGH_REWARD_OPTION_ALL_FILE = f"{PROJECT_BEH_ALL_DATA_DIR}/NoNanBehAll_RelIrrelHighReward.csv"
# behavioral data for both read and write, inclduing relevant and irrelevant high reward options, avegrage across phases
PROJECT_NoNAN_BEH_REL_IRREL_HIGH_REWARD_OPTION_GROUPBY_ALL_FILE = f"{PROJECT_BEH_ALL_DATA_DIR}/NoNanBehAll_RelIrrelHighReward_Groupby.csv"
# behavioral data for both read and write, inclduing relevant and irrelevant high reward options, avegrage across phases with summary of model parameter weights
PROJECT_NoNAN_BEH_REL_IRREL_HIGH_REWARD_OPTION_GROUPBY_ALL_FILE_MODEL_PARAMETER = f"{PROJECT_BEH_ALL_DATA_DIR}/NoNanBehAll_RelIrrelHighReward_Groupby_ModelParameter.csv"
########### model fit setting
# Number of chains in MCMC procedure
N_CHAIN = 8
# The number of iteration or samples for each chain in MCM procedure
N_SAMPLES=3000
# number of warp up samples
N_WARMUP = 1000


########### List of Participants
PARTICIPANTS_PD = ['sub-004', 'sub-010', 'sub-041', 'sub-005', 'sub-040', 'sub-029',
                   'sub-045', 'sub-042', 'sub-052', 'sub-059', 'sub-025', 'sub-056',
                   'sub-065', 'sub-070', 'sub-071', 'sub-074', 'sub-082', 'sub-085',
                   'sub-086', 'sub-087', 'sub-089', 'sub-092', 'sub-108', 'sub-109']
PARTICIPANTS_HC =['sub-064', 'sub-077', 'sub-079', 'sub-080', 'sub-088', 'sub-121',
                  'sub-012', 'sub-036', 'sub-026', 'sub-034', 'sub-033', 'sub-044',
                  'sub-030', 'sub-047', 'sub-054', 'sub-048', 'sub-067', 'sub-060',
                  'sub-069', 'sub-075', 'sub-078', 'sub-081', 'sub-090']
PARTICIPANTS = ['sub-004', 'sub-010', 'sub-041', 'sub-005', 'sub-040', 'sub-029',
                'sub-045', 'sub-042', 'sub-052', 'sub-059', 'sub-025', 'sub-056',
                'sub-065', 'sub-070', 'sub-071', 'sub-074', 'sub-082', 'sub-085',
                'sub-086', 'sub-087', 'sub-089', 'sub-092', 'sub-108', 'sub-109',
                'sub-064', 'sub-077', 'sub-079', 'sub-080', 'sub-088', 'sub-121',
                'sub-012', 'sub-036', 'sub-026', 'sub-034', 'sub-033', 'sub-044',
                'sub-030', 'sub-047', 'sub-054', 'sub-048', 'sub-067', 'sub-060',
                'sub-069', 'sub-075', 'sub-078', 'sub-081', 'sub-090']