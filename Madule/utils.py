import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os


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
