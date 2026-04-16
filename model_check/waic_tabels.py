#!/mrhome/amingk/anaconda3/envs/7tpd/bin/python
import numpy as np
import pandas as pd
import stan
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/mrhome/amingk/Documents/7TPD/ActStimRL')
import arviz as az
from scipy import stats
from utils import *

# model list in tabel1
#table1_models = ['tabel1_model1', 'tabel1_model2', 'tabel1_model3', 
#                 'tabel1_model4', 'tabel1_model5']
#waic_models(model_calss='table1', list_model=table1_models)

# model list in tabel2
table2_models = ['tabel2_model1', 'tabel2_model2', 'tabel2_model3', 
                 'tabel2_model4', 'tabel2_model5', 'tabel2_model6']
waic_models(model_calss='tabel2', list_model=table2_models)
