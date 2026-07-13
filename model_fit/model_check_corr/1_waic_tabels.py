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

# model list in tabel2
table2_models = ['model1', 'model2', 'model3', 
                 'model4', 'model5']
waic_models(model_calss='tabel1', list_model=table2_models)
