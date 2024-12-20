import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt 


"""Plots of choice responses for each individuals.
In this plot all feasturs of pushing/pulling and yellow chosen/ blue chosen are shown in one plot.
The both choices and rewarded option Are seperated for each feature.
At first you should run 'pool-data.py' to create '_achieva7t_task-DA_beh.csv' for each participant and then use
the current madule to plot choice and rewarded options with together for each participant.
"""
 

def plotChosenCorrect(data, blocks, subName, saveFile):
    """Plot of chosen and correct response for all runs and sessions"""
    # Figure of behavioral data in two column and four rows
    mm = 1/2.54  # centimeters in inches
    fig = plt.figure(figsize=(20*mm, 5*mm), tight_layout=True)
    rows = 1
    columns = 2
    # Position marker type and colors of Action adn Color Value Learning
    y = [.8 ,.7, .5 ,.3, .2] 
    markers = ['v', 'o', 'o' , 'o', '^']
    colorsAct =['#2ca02c','#2ca02c', '#d62728', '#9467bd', '#9467bd']
    colorsClr =['#bcbd22','#bcbd22', '#d62728', '#1f77b4', '#1f77b4']
    
    # session 1
    s = 1
    # run 2
    r = 2

    for idx in range(0, 2):
        fig.add_subplot(rows, columns, idx+1) 
        # Action block
        if blocks[idx] == 'Act':
            # Seperate data taken from a session, run and Action block
            dataCondAct = data[(data.session==s) & (data.run==r) & (data.block==blocks[idx])]
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
            plt.yticks([])
            plt.xticks([1,10,20, 30, 42])
            plt.ylim(.15, .85)
            plt.xlabel('Trials', fontsize=12)
            if blocks[idx]=='Stim':
                blockName = 'Clr'
            elif blocks[idx]=='Act':
                blockName = 'Act'
                
            plt.title(subName + ' - Ses ' +  str(s) +' - Run ' + str(r) + ' - ' +  blockName + ' Value Learning' , fontsize=10)    
            plt.legend(dfPlotAct.label, fontsize=5)      
        # Color block
        elif blocks[idx] == 'Stim':
            # Seperate data taken from a session, run and Color block
            dataCondClr = data[(data.session==s) & (data.run==r) & (data.block==blocks[idx])]
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
            plt.yticks([])
            plt.xticks([1,10,20, 30, 42])
            plt.ylim(.15, .85)
            plt.xlabel('Trials', fontsize=12) 
            if blocks[idx]=='Stim':
                blockName = 'Clr'
            elif blocks[idx]=='Act':
                blockName = 'Act'
                
            plt.title(subName + ' - Ses ' +  str(s) +' - Run ' + str(r) + ' - ' +  blockName + ' Value Learning' , fontsize=10)    
            plt.legend(dfPlotClr.label, fontsize=5) 
            
        # Determine Reversal point for each condition 
        reverse = data.loc[(data['session']==s)&(data['run']==r)&(data['block']==blocks[idx]), 'reverse'].unique()[0]
        # Draw vertical lines for one or two reversal points learning during runs
        if reverse==21:
            plt.axvline(x = 21, color='#ff7f0e', linewidth=1, alpha=.5)
        elif reverse==14:
            plt.axvline(x = 14, color='#ff7f0e', linewidth=1, alpha=.7)
            plt.axvline(x = 28, color='#ff7f0e', linewidth=1, alpha=.7)

    # Save plot of chosen and correct response 
    fig.savefig(saveFile, dpi=300)
    plt.close()
    
subName = 'sub-012'
# Main directory of the subject
subMainDirec = '/mnt/projects/7TPD/bids/derivatives/fMRI_DA/Amin/BehData/'
# Directory of main dataset for each participant
dirc = subMainDirec + subName + '/' + subName + '_achieva7t_task-DA_beh.csv'
# Read the excel file
data = pd.read_csv(dirc)
# Condition sequences for each particiapnt
blocks = data.groupby(['session', 'run'])['block'].unique().to_numpy()
blocks = np.array(blocks[1]).flatten()
#save file name
saveFile = f'../../Figures/{subName}_plot_choice_reward_option.png'
# Plot by a pre implemented madule
plotChosenCorrect(data = data, blocks = blocks, subName = subName, saveFile = saveFile)
