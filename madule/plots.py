import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

def plotChosenCorrect(data, subName, saveFile):
    """Plot of chosen and correct response for all runs and sessions"""
    # First type of learning in sesion 1 and run 1    
    firstActOrClr = data.iloc[0]['stimActFirst']
    # Figure of behavioral data in two column and four rows
    fig = plt.figure(figsize=(10, 10), tight_layout=True)
    rows = 4
    columns = 2
    # Positionm marker type and colors of Action adn Color Value Learning
    y = [3.2 ,3, 2.1 , 1.2, 1] 
    markers = ['v', 'o', 'o' , 'o', '^']
    colorsAct =['#2ca02c','#2ca02c', '#d62728', '#9467bd', '#9467bd']
    colorsClr =['#bcbd22','#bcbd22', '#d62728', '#1f77b4', '#1f77b4']
    # List of session and run for Action First in Session 1
    titlesAct = [subName + ' - Ses 1 - Run 1 - Act Learning', 
                 subName + ' - Ses 1 - Run 1 - Clr Learning',
                 subName + ' - Ses 1 - Run 2 - Clr Learning',
                 subName + ' - Ses 1 - Run 2 - Act Learning',
                 subName + ' - Ses 2 - Run 1 - Clr Learning', 
                 subName + ' - Ses 2 - Run 1 - Act Learning',
                 subName + ' - Ses 2 - Run 2 - Act Learning',
                 subName + ' - Ses 2 - Run 2 - Clr Learning']
    # List of session and run for Stimulus First in Session 1
    titlesClr = [subName + ' - Ses 1 - Run 1 - Clr Learning', 
                 subName + ' - Ses 1 - Run 1 - Act Learning',
                 subName + ' - Ses 1 - Run 2 - Act Learning',
                 subName + ' - Ses 1 - Run 2 - Clr Learning',
                 subName + ' - Ses 2 - Run 1 - Act Learning', 
                 subName + ' - Ses 2 - Run 1 - Clr Learning',
                 subName + ' - Ses 2 - Run 2 - Clr Learning',
                 subName + ' - Ses 2 - Run 2 - Act Learning']
    # Order of Action and Stimulus for sessions and runs
    orderAct = ['Act', 'Stim', 'Stim', 'Act', 'Stim', 'Act', 'Act', 'Stim']
    orderClr = ['Stim', 'Act', 'Act', 'Stim', 'Act', 'Stim', 'Stim', 'Act']
    if firstActOrClr == 'Act':
        titles = titlesAct
        order = orderAct
    elif firstActOrClr=='Stim':
        titles = titlesClr
        order = orderClr

    idx = 0
    for s in range(1, 3):
        for r in range(1, 3):
            for b in range(1, 3):
                fig.add_subplot(rows, columns, idx+1) 
                # Action block
                if order[idx] == 'Act':
                    # Seperate data taken from a session, run and Action block
                    dataCondAct = data[(data.session==s) & (data.run==r) & (data.block==order[idx])]
                    # Seperate the index of pushed and pulled responses
                    resAct = dataCondAct['pushed'].to_numpy().astype(int)
                    pushed = np.where(resAct==1)[0] + 1
                    pulled = np.where(resAct==0)[0] + 1
                    noRes  = np.where(resAct < 0)[0] + 1
                    # Seperate the index of pushed and pulled correct choices
                    corrAct= dataCondAct['pushCorrect']
                    pushCorr = np.where(corrAct==1)[0] + 1
                    pulledCorr = np.where(corrAct==0)[0] + 1
                    # Put all reponses and correct choice in a Dataframe
                    dicDataAct = ({'label': ['pushed', 'push correct', 'no response', 'pull correct', 'pulled'],
                                'x': [pushed, pushCorr, noRes, pulledCorr, pulled]})
                    dfPlotAct = pd.DataFrame(dicDataAct)
                    # Create a list of y coordinates for every x coordinate
                    for i in range(len(dfPlotAct)):
                        plt.scatter(dfPlotAct.x[i],[y[i] for j in range(len(dfPlotAct.x[i]))], 
                                    s=10, c=colorsAct[i], marker=markers[i])
                    # show the empy y axis label
                    plt.yticks(y,[]) 
                    plt.xlabel('Trials', fontsize=12)
                    plt.title(titles[idx], fontsize=10)    
                    plt.legend(dfPlotAct.label, fontsize=8)      
                # Color block
                elif order[idx] == 'Stim':
                    # Seperate data taken from a session, run and Color block
                    dataCondClr = data[(data.session==s) & (data.run==r) & (data.block==order[idx])]
                    # Seperate the index of yellow and blue responses
                    resClr = dataCondClr['yellowChosen'].to_numpy().astype(int)
                    yellChosen = np.where(resClr==1)[0] + 1
                    blueChosen = np.where(resClr==0)[0] + 1
                    noRes  = np.where(resClr < 0)[0] + 1
                    # Seperate the index of yellow and blue correct choices
                    corrClr= dataCondClr['yellowCorrect']
                    yellCorr = np.where(corrClr==1)[0] + 1
                    blueCorr = np.where(corrClr==0)[0] + 1
                    # Put all reponses and correct choice in a Dataframe
                    dicDataClr = ({'label': ['yellow chosen', 'yellow correct', 'no response', 'blue correct', 'blue chosen'],
                                'x': [yellChosen, yellCorr, noRes, blueCorr, blueChosen]})
                    dfPlotClr = pd.DataFrame(dicDataClr)         
                    #create a list of y coordinates for every x coordinate
                    for i in range(len(dfPlotClr)):
                        plt.scatter(dfPlotClr.x[i],[y[i] for j in range(len(dfPlotClr.x[i]))], 
                                    s=10, c=colorsClr[i], marker=markers[i])
                    # Show the empy y axis label
                    plt.yticks(y,[]) 
                    plt.xlabel('Trials', fontsize=12) 
                    plt.title(titles[idx], fontsize=10)   
                    plt.legend(dfPlotClr.label, fontsize=8)          
                # Draw vertical lines for one or two reversal points learning during runs
                if idx%2==1:
                    plt.axvline(x = 21, color='#ff7f0e', linewidth=1, alpha=.5)
                else:
                    plt.axvline(x = 14, color='#ff7f0e', linewidth=1, alpha=.7)
                    plt.axvline(x = 28, color='#ff7f0e', linewidth=1, alpha=.7)

                idx += 1
    # Save plot of chosen and correct response 
    fig.savefig(saveFile + '.png', dpi=300)
     
# Taken from https://github.com/laurafontanesi/rlssm/blob/main/rlssm/utils.py 
def bci(x, alpha=0.05):
    """Calculate Bayesian credible interval (BCI).
    Parameters
    ----------
    x : array-like
        An array containing MCMC samples.
    alpha : float
        Desired probability of type I error (defaults to 0.05).
    Returns
    -------
    interval : numpy.ndarray
        Array containing the lower and upper bounds of the bci interval.
    """

    interval = np.nanpercentile(x, [(alpha/2)*100, (1-alpha/2)*100])

    return interval

def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of a given width.
    Parameters
    ----------
    x : array-like
        An sorted numpy array.
    alpha : float
        Desired probability of type I error (defaults to 0.05).
    Returns
    -------
    hdi_min : float
        The lower bound of the interval.
    hdi_max : float
        The upper bound of the interval.
    """

    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max


def hdi(x, alpha=0.05):
    """Calculate highest posterior density (HPD).
        Parameters
        ----------
        x : array-like
            An array containing MCMC samples.
        alpha : float
            Desired probability of type I error (defaults to 0.05).
    Returns
    -------
    interval : numpy.ndarray
        Array containing the lower and upper bounds of the hdi interval.
    """

    # Make a copy of trace
    x = x.copy()
     # Sort univariate node
    sx = np.sort(x)
    interval = np.array(calc_min_interval(sx, alpha))

    return interval

def plot_posterior(x,
                   ax=None,
                   gridsize=100,
                   clip=None,
                   show_intervals="HDI",
                   alpha_intervals=.05,
                   color='grey',
                   intervals_kws=None,
                   trueValue = None,
                   title = None,
                   xlabel = None,
                   ylabel = None,
                   legends = None,
                   **kwargs):
    """Plots a univariate distribution with Bayesian intervals for inference.

    By default, only plots the kernel density estimation using scipy.stats.gaussian_kde.

    Bayesian instervals can be also shown as shaded areas,
    by changing show_intervals to either BCI or HDI.

    Parameters
    ----------

    x : array-like
        Usually samples from a posterior distribution.

    ax : matplotlib.axes.Axes, optional
        If provided, plot on this Axes.
        Default is set to current Axes.

    gridsize : int, default to 100
        Resolution of the kernel density estimation function.

    clip : tuple of (float, float), optional
        Range for the kernel density estimation function.
        Default is min and max values of `x`.

    show_intervals : str, default to "HDI"
        Either "HDI", "BCI", or None.
        HDI is better when the distribution is not simmetrical.
        If None, then no intervals are shown.

    alpha_intervals : float, default to .05
        Alpha level for the intervals calculation.
        Default is 5 percent which gives 95 percent BCIs and HDIs.

    intervals_kws : dict, optional
        Additional arguments for `matplotlib.axes.Axes.fill_between`
        that shows shaded intervals.
        By default, they are 50 percent transparent.

    color : matplotlib.colors
        Color for both the density curve and the intervals.

    Returns
    -------

    ax : matplotlib.axes.Axes
        Returns the `matplotlib.axes.Axes` object with the plot
        for further tweaking.

    """
    if clip is None:
        min_x = np.min(x)
        max_x = np.max(x)
    else:
        min_x, max_x = clip

    if ax is None:
        ax = plt.gca()
        
    if trueValue is not None:
        ax.axvline(x=trueValue, ls='--')

    if intervals_kws is None:
        intervals_kws = {'alpha':.5}

    density = gaussian_kde(x, bw_method='scott')
    xd = np.linspace(min_x, max_x, gridsize)
    yd = density(xd)

    ax.plot(xd, yd, color=color, **kwargs)

    if show_intervals is not None:
        if np.sum(show_intervals == np.array(['BCI', 'HDI'])) < 1:
            raise ValueError("must be either None, BCI, or HDI")
        if show_intervals == 'BCI':
            low, high = bci(x, alpha_intervals)
        else:
            low, high = hdi(x, alpha_intervals)
        ax.fill_between(xd[np.logical_and(xd >= low, xd <= high)],
                        yd[np.logical_and(xd >= low, xd <= high)],
                        color=color,
                        **intervals_kws)
    
    if legends is not None:
        ax.legend(legends) 
        
    if title is not None:
        plt.title(title, fontsize=12)

    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=12)
    
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=14)
    
   
    sns.despine()
     
    return ax    