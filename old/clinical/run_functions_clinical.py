#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python

import numpy as np 
import pandas as pd
import stan
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
import os
from utils import model_utils
from utils import config

# calculated MAP posterior paramters
model_utils.compute_and_save_clinical_parameters(readModel=config.PROJECT_HIER_MODEL_DIR)