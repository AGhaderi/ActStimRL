import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.lines as mlines

def plotChosenCorrect(data, blocks, subName, saveFile):
    """Plot of chosen and correct response for all runs and sessions"""
    # Figure of behavioral data in two column and four rows
    fig = plt.figure(figsize=(10, 10), tight_layout=True)
    rows = 4
    columns = 2
    # Position marker type and colors of Action adn Color Value Learning
    y = [1.2 ,1.1, .6 ,.2, .1] 
    markers = ['v', 'o', 'o' , 'o', '^']
    colorsAct =['#2ca02c','#2ca02c', '#d62728', '#9467bd', '#9467bd']
    colorsClr =['#bcbd22','#bcbd22', '#d62728', '#1f77b4', '#1f77b4']
    
    idx = 0
    for s in range(1, 3):
        for r in range(1, 3):
            for b in range(1, 3):
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
                    plt.xlabel('Trials', fontsize=12)
                    if blocks[idx]=='Stim':
                        blockName = 'Clr'
                    elif blocks[idx]=='Act':
                        blockName = 'Act'
                        
                    plt.title(subName + ' - Ses ' +  str(s) +' - Run ' + str(r) + ' - ' +  blockName + ' Value Learning' , fontsize=10)    
                    plt.legend(dfPlotAct.label, fontsize=8)      
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
                    plt.xlabel('Trials', fontsize=12) 
                    plt.xlabel('Trials', fontsize=12)
                    if blocks[idx]=='Stim':
                        blockName = 'Clr'
                    elif blocks[idx]=='Act':
                        blockName = 'Act'
                        
                    plt.title(subName + ' - Ses ' +  str(s) +' - Run ' + str(r) + ' - ' +  blockName + ' Value Learning' , fontsize=10)    
                    plt.legend(dfPlotClr.label, fontsize=8) 
                    
                # Determine Reversal point for each condition 
                reverse = data.loc[(data['session']==s)&(data['run']==r)&(data['block']==blocks[idx]), 'reverse'].unique()[0]
                # Draw vertical lines for one or two reversal points learning during runs
                if reverse==21:
                    plt.axvline(x = 21, color='#ff7f0e', linewidth=1, alpha=.5)
                elif reverse==14:
                    plt.axvline(x = 14, color='#ff7f0e', linewidth=1, alpha=.7)
                    plt.axvline(x = 28, color='#ff7f0e', linewidth=1, alpha=.7)

                idx += 1
    # Save plot of chosen and correct response 
    fig.savefig(saveFile, dpi=300)
    plt.close()
     
def plotChosenCorrect_modofied1(data, blocks, subName, saveFile):
    """Plot of chosen and correct response for all runs and sessions.
       In this plot we combine both correct option and choices """
    # Figure of behavioral data in two column and four rows
    fig = plt.figure(figsize=(10, 10), tight_layout=True)
    rows = 8
    columns = 1
    # Position marker type and colors of Action adn Color Value Learning
    y = [1 ,1, .5 ,.0, .0] 
    markers = ['$\u25EF$', 'o', 'o' , 'o', '$\u25EF$']
    colors =['#d62728','#2ca02c', '#d62728', '#1f77b4', '#d62728']
    size = [30, 5, 10, 10, 30]
    idx = 0
    for s in range(1, 3):
        for r in range(1, 3):
            for b in range(1, 3):
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
                                    s=size[i], facecolors=colors[i], marker=markers[i])
                    # show the empy y axis label
                    plt.yticks(y,[]) 
                    plt.ylim(-.1, 1.1)
                    plt.xlabel('')
                    if blocks[idx]=='Stim':
                        blockName = 'Clr'
                    elif blocks[idx]=='Act':
                        blockName = 'Act'
                        
                    plt.title(subName + ' - Ses ' +  str(s) +' - Run ' + str(r) + ' - ' +  blockName + ' Value Learning' , fontsize=10)    
                    plt.legend(['Chosen', 'pull correct', 'no response', 'pull correct'], fontsize=5)      
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
                                    s=size[i], facecolors=colors[i], marker=markers[i])
                    # Show the empy y axis label
                    plt.yticks(y,[]) 
                    plt.ylim(-.1, 1.1)
                    plt.xlabel('') 
                    if blocks[idx]=='Stim':
                        blockName = 'Clr'
                    elif blocks[idx]=='Act':
                        blockName = 'Act'
                        
                    plt.title(subName + ' - Ses ' +  str(s) +' - Run ' + str(r) + ' - ' +  blockName + ' Value Learning' , fontsize=10)    
                    plt.legend(['Chosen', 'yellow correct', 'no response', 'blue correct'], fontsize=5) 
                    
                # Determine Reversal point for each condition 
                reverse = data.loc[(data['session']==s)&(data['run']==r)&(data['block']==blocks[idx]), 'reverse'].unique()[0]
                # Draw vertical lines for one or two reversal points learning during runs
                if reverse==21:
                    plt.axvline(x = 21, color='#ff7f0e', linewidth=1, alpha=.5)
                elif reverse==14:
                    plt.axvline(x = 14, color='#ff7f0e', linewidth=1, alpha=.7)
                    plt.axvline(x = 28, color='#ff7f0e', linewidth=1, alpha=.7)

                idx += 1
    
    plt.xlabel('Trials', fontsize=12)
    # Save plot of chosen and correct response 
    fig.savefig(saveFile, dpi=300)
    plt.close()
     
def plotChosenCorrect_modofied2(data, blocks, subName, saveFile):
    """Plot of chosen and correct response for all runs and sessions.
       In this plot we combine both correct option and choices and also use the amount of options"""
    # Figure of behavioral data in two column and four rows
    fig = plt.figure(figsize=(10, 10), tight_layout=True)
    rows = 8
    columns = 1
    # Position marker type and colors of Action adn Color Value Learning
    y = [1 ,1, .5 ,.0, .0] 
    markers = ['$\u25EF$', 'o', 'o' , 'o', '$\u25EF$']
    colors =['#d62728','#2ca02c', '#d62728', '#1f77b4', '#d62728']
    idx = 0
    for s in range(1, 3):
        for r in range(1, 3):
            for b in range(1, 3):
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
                    # amount of correct option 
                    amtPushCorr = dataCondAct[dataCondAct['pushCorrect']==1]['winAmtPushable'].to_numpy()
                    amtPullCorr = dataCondAct[dataCondAct['pushCorrect']==0]['winAmtPullable'].to_numpy()
                    # changing size of chosen option adjusted by the winning amount
                    size = [100*np.ones(pushed.shape[0]), amtPushCorr/4, 10*np.ones(noRes.shape[0]),
                             amtPullCorr/4, 100*np.ones(pulled.shape[0])]
                    # Create a list of y coordinates for every x coordinate
                    for i in range(len(dfPlotAct)):
                        plt.scatter(dfPlotAct.x[i],[y[i] for j in range(len(dfPlotAct.x[i]))], 
                                    s=size[i], facecolors=colors[i], marker=markers[i])
                    # show the empy y axis label
                    plt.yticks(y,[]) 
                    plt.ylim(-.15, 1.15)
                    plt.xlabel('')
                    if blocks[idx]=='Stim':
                        blockName = 'Clr'
                    elif blocks[idx]=='Act':
                        blockName = 'Act'
                        
                    plt.title(subName + ' - Ses ' +  str(s) +' - Run ' + str(r) + ' - ' +  blockName + ' Value Learning' , fontsize=10)    
                    plt.legend(['Chosen', 'push correct', 'no response', 'blue correct'], fontsize=5)      
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
                    # amount of correct option 
                    amtYellCorr = dataCondClr[dataCondClr['yellowCorrect']==1]['winAmtYellow'].to_numpy()
                    amtBlueCorr = dataCondClr[dataCondClr['yellowCorrect']==0]['winAmtBlue'].to_numpy()
                    # changing size of chosen option adjusted by the winning amount
                    size = [100*np.ones(yellChosen.shape[0]), amtYellCorr/4, 10*np.ones(noRes.shape[0]),
                             amtBlueCorr/4, 100*np.ones(blueChosen.shape[0])]
                    #create a list of y coordinates for every x coordinate
                    for i in range(len(dfPlotClr)):
                        plt.scatter(dfPlotClr.x[i],[y[i] for j in range(len(dfPlotClr.x[i]))], 
                                    s=size[i], facecolors=colors[i], marker=markers[i])
                    # Show the empy y axis label
                    plt.yticks(y,[]) 
                    plt.ylim(-.15, 1.15)
                    plt.xlabel('') 
                    if blocks[idx]=='Stim':
                        blockName = 'Clr'
                    elif blocks[idx]=='Act':
                        blockName = 'Act'
                        
                    plt.title(subName + ' - Ses ' +  str(s) +' - Run ' + str(r) + ' - ' +  blockName + ' Value Learning' , fontsize=10)    
                    plt.legend(['Chosen', 'yellow correct', 'no response', 'blue correct'], fontsize=4) 
                    
                # Determine Reversal point for each condition 
                reverse = data.loc[(data['session']==s)&(data['run']==r)&(data['block']==blocks[idx]), 'reverse'].unique()[0]
                # Draw vertical lines for one or two reversal points learning during runs
                if reverse==21:
                    plt.axvline(x = 21, color='#ff7f0e', linewidth=1, alpha=.5)
                elif reverse==14:
                    plt.axvline(x = 14, color='#ff7f0e', linewidth=1, alpha=.7)
                    plt.axvline(x = 28, color='#ff7f0e', linewidth=1, alpha=.7)

                idx += 1
    plt.xlabel('Trials', fontsize=12)
    # Save plot of chosen and correct response 
    fig.savefig(saveFile, dpi=300)
    plt.close()

def plotChosenCorrect_modofied3(data, blocks, subName, saveFile):
    """Plot of chosen and correct response for all runs and sessions.
       In this plot we combine both correct option and choices """
    # Figure of behavioral data in two column and four rows
    fig = plt.figure(figsize=(10, 10), tight_layout=True)
    rows = 8
    columns = 1
    # Position marker type and colors of Action adn Color Value Learning
    y = [1 ,1, .5 ,.0, .0] 
    markers = ['$\u25EF$', 'o', 'o' , 'o', '$\u25EF$']
    size = [100, 15, 5, 15, 100]
    idx = 0
    for s in range(1, 3):
        for r in range(1, 3):
            for b in range(1, 3):
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
                    # changing color of chosen option adjusted by the higher vs lower winning amount
                    # Calculate the probability of high amount is chosed or lower amount
                    amtPushCorr = dataCondAct[resAct==1]['winAmtPushable'].to_numpy()
                    amtPullCorr = dataCondAct[resAct==0]['winAmtPullable'].to_numpy()
                    colors = [[i*'#d62728' + j*'#d6d327' for (i,j) in zip(amtPushCorr>=50, amtPushCorr<50)], 
                              pushCorr.shape[0]*['#2ca02c'], 
                              noRes.shape[0]*['#d62728'],
                             pulledCorr.shape[0]*['#1f77b4'], 
                             [i*'#d62728' + j*'#d6d327' for (i,j) in zip(amtPullCorr>=50, amtPullCorr<50)]]
                    # Create a list of y coordinates for every x coordinate
                    for i in range(len(dfPlotAct)):
                        plt.scatter(dfPlotAct.x[i],[y[i] for j in range(len(dfPlotAct.x[i]))], 
                                    s=size[i], facecolors=colors[i], marker=markers[i])
                    # show the empy y axis label
                    plt.yticks(y,[]) 
                    plt.ylim(-.15, 1.15)
                    plt.xlabel('')
                    if blocks[idx]=='Stim':
                        blockName = 'Clr'
                    elif blocks[idx]=='Act':
                        blockName = 'Act'

                    plt.title(subName + ' - Ses ' +  str(s) +' - Run ' + str(r) + ' - ' +  blockName + ' Value Learning' , fontsize=10) 
                    # chaning legen manually since fro each section of push and pull has two higher and lowe amount so
                    # it is hard to disantagle that automatically, of course based on my implementation
                    high_amt = mlines.Line2D([], [], color='#d62728', marker='$\u25EF$', ls='', markersize=6, label='high amt')
                    low_amt = mlines.Line2D([], [], color='#d6d327', marker='$\u25EF$', ls='', markersize=6, label='low amt')
                    no_reponse = mlines.Line2D([], [], color='#2ca02c', marker='o', ls='', markersize=2, label='push correct')
                    push_correct = mlines.Line2D([], [], color='#d62728', marker='o', ls='', markersize=2, label='no response')
                    pull_correct = mlines.Line2D([], [], color='#1f77b4', marker='o', ls='', markersize=2, label='pull correct')
                    plt.legend(handles=[high_amt, low_amt, no_reponse, push_correct, pull_correct], fontsize=6)
                        
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
                    # Calculate the probability of high amount is chosed or lower amount
                    # (resClr >= 0) this is for elinination of non response choices
                    amtYellCorr = dataCondClr[resClr == 1]['winAmtYellow'].to_numpy()
                    amtBlueCorr = dataCondClr[resClr == 0]['winAmtBlue'].to_numpy()                    
                    colors = [[i*'#d62728' + j*'#d6d327' for (i,j) in zip(amtYellCorr>=50, amtYellCorr<50)], 
                              yellCorr.shape[0]*['#2ca02c'], 
                              noRes.shape[0]*['#d62728'],
                              blueCorr.shape[0]*['#1f77b4'], 
                             [i*'#d62728' + j*'#d6d327' for (i,j) in zip(amtBlueCorr>=50, amtBlueCorr<50)]]
                    #create a list of y coordinates for every x coordinate
                    for i in range(len(dfPlotClr)):
                        plt.scatter(dfPlotClr.x[i],[y[i] for j in range(len(dfPlotClr.x[i]))], 
                                    s=size[i], facecolors=colors[i], marker=markers[i])
                    # Show the empy y axis label
                    plt.yticks(y,[]) 
                    plt.ylim(-.15, 1.15)
                    plt.xlabel('') 
                    if blocks[idx]=='Stim':
                        blockName = 'Clr'
                    elif blocks[idx]=='Act':
                        blockName = 'Act'
                    plt.title(subName + ' - Ses ' +  str(s) +' - Run ' + str(r) + ' - ' +  blockName + ' Value Learning' , fontsize=10)    
                    # chaning legen manually since fro each section of push and pull has two higher and lowe amount so
                    # it is hard to disantagle that automatically, of course based on my implementation
                    high_amt = mlines.Line2D([], [], color='#d62728', marker='$\u25EF$', ls='', markersize=6, label='high amt')
                    low_amt = mlines.Line2D([], [], color='#d6d327', marker='$\u25EF$', ls='', markersize=6, label='low amt')
                    no_reponse = mlines.Line2D([], [], color='#2ca02c', marker='o', ls='', markersize=2, label='yellow correct')
                    push_correct = mlines.Line2D([], [], color='#d62728', marker='o', ls='', markersize=2, label='no response')
                    pull_correct = mlines.Line2D([], [], color='#1f77b4', marker='o', ls='', markersize=2, label='blue correct')
                    plt.legend(handles=[high_amt, low_amt, no_reponse, push_correct, pull_correct], fontsize=6)

                # Determine Reversal point for each condition 
                reverse = data.loc[(data['session']==s)&(data['run']==r)&(data['block']==blocks[idx]), 'reverse'].unique()[0]
                # Draw vertical lines for one or two reversal points learning during runs
                if reverse==21:
                    plt.axvline(x = 21, color='#ff7f0e', linewidth=1, alpha=.5)
                elif reverse==14:
                    plt.axvline(x = 14, color='#ff7f0e', linewidth=1, alpha=.7)
                    plt.axvline(x = 28, color='#ff7f0e', linewidth=1, alpha=.7)

                idx += 1
    
    plt.xlabel('Trials', fontsize=12)
    # Save plot of chosen and correct response 
    fig.savefig(saveFile, dpi=300)
    plt.close()
     

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

def plot_posterior(data,
                   gridsize=100,
                   clip=None,
                   show_intervals="HDI",
                   alpha_intervals=.05,
                   color='grey',
                   intervals_kws=None,
                   xlabel=None,
                   ylabel=None,
                   title=None,
                   legend=None):
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
    fig = plt.figure(figsize=(6,4))
    data = data.reshape(data.shape[0],-1)
    for i in range(data.shape[1]):
        x = data[:,i]
        if clip is None:
            min_x = np.min(x)
            max_x = np.max(x)
        else:
            min_x, max_x = clip

        if intervals_kws is None:
            intervals_kws = {'alpha':.5}

        density = gaussian_kde(x, bw_method='scott')
        xd = np.linspace(min_x, max_x, gridsize)
        yd = density(xd)

        plt.plot(xd, yd, color=color[i])

        if show_intervals is not None:
            if np.sum(show_intervals == np.array(['BCI', 'HDI'])) < 1:
                raise ValueError("must be either None, BCI, or HDI")
            if show_intervals == 'BCI':
                low, high = bci(x, alpha_intervals)
            else:
                low, high = hdi(x, alpha_intervals)
            plt.fill_between(xd[np.logical_and(xd >= low, xd <= high)],
                            yd[np.logical_and(xd >= low, xd <= high)],
                            color=color[i],
                            **intervals_kws) 
   
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if legend is not None:
        fig.legend(labels=legend)
    sns.despine()