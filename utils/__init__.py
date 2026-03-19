"""
A collection of utility functions for plotting, simulation ans other supporting madules.
""" 

from .config import (PROJECT_DATA_DIR, 
                     PROJECT_BEH_ALL_DATA_DIR, 
                     FIGURES_DIR,  OUTPUT_CSV_DIR, 
                     PROJECT_REDCAP_DIR, PROJECT_CLIN_EVAL_DIR,
                     PROJECT_HIER_MODEL_DIR, STAN_DIR, 
                     SCRATCH_BEH_ALL_DATA_DIR , 
                     SCRATCH_CLIN_EVAL_DIR, 
                     SCRATCH_HIER_MODEL_DIR, 
                     COLORS, OPACITY, 
                     PROJECT_RAW_BEH_ALL_FILE, 
                     PROJECT_NoNAN_BEH_ALL_FILE,  
                     PROJECT_CLIN_EVAL_FILE, 
                     PROJECT_MAP_CLIN_EVAL_FILE, 
                     PROJECT_NoNAN_BEH_REL_IRREL_HIGH_REWARD_OPTION_ALL_FILE, 
                     PROJECT_NoNAN_BEH_REL_IRREL_HIGH_REWARD_OPTION_GROUPBY_ALL_FILE, 
                     N_CHAIN, 
                     N_SAMPLES, 
                     N_WARMUP)


from .plotting import (plotRelevantOptionTrial,
                       plotIrrelevantOptionTrial,
                       plotChoiceResponse,
                       plotFeatureBias,
                       plotProportionRelIrrelevantHighRewardOption,
                       plot_posterior)

from .data_utils import (build_behavior_dataframe,
                         calRelevantAndIrrelevantHighRewardOptionTrial)

from .model_utils import (compute_and_save_clinical_parameters,
                         dataStanActClr,
                         to_pickle,
                         load_pickle,
                         waic)

from .random import (simStrategyBehavior)
 