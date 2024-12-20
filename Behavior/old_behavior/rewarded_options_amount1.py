"""Intrinsic tendency of reward probability toward left/ right, push/pull, yellow/blue across participant for each condition."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""Rewarded biad based on the task"""
# read collected data across data
rawBehAll = behAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/Amin/BehData/AllBehData/rawBehAll.csv')
# gtoup label 1,2 3 are PD-OFF, HC and PF-ON respectively
rawBehAll['group'] = rawBehAll['group'].replace([1,2,3], ['PD-OFF', 'HC', 'PD-ON'])
# gtoup label 1,2 3 are PD-OFF, HC and PF-ON respectively
rawBehAll['Condition'] = rawBehAll['block'].replace(['Act', 'Stim'], ['Act', 'Clr'])

# Find chosen amount for each trial
chosenAmount = rawBehAll['leftChosen']*rawBehAll['winAmtLeft'] + (1-rawBehAll['leftChosen'])*rawBehAll['winAmtRight'] 
# Calculate the probability of high amount is chosed or lower amount
rawBehAll['chosenHighWinAmt'] = chosenAmount>=50
# Calculuate reward left option
leftCorrect = rawBehAll['leftCanBePushed                ']*rawBehAll.pushCorrect + (1-rawBehAll['leftCanBePushed                '])*(1-rawBehAll.pushCorrect)
rawBehAll['leftCorrect'] = leftCorrect

# the proportion of rewarded left, push and chosen yellow  
leftCorrect = rawBehAll[rawBehAll['chosenHighWinAmt']==1].groupby(['group', 'Condition', 'sub_ID'], as_index=False)['leftCorrect'].mean()
pushCorrect = rawBehAll[rawBehAll['chosenHighWinAmt']==1].groupby(['group', 'Condition', 'sub_ID'], as_index=False)['pushCorrect'].mean()
yellowCorrect = rawBehAll[rawBehAll['chosenHighWinAmt']==1].groupby(['group', 'Condition', 'sub_ID'], as_index=False)['yellowCorrect'].mean()

# plot of probability reward in the task design
mm = 1/2.54  # centimeters in inches
fig = plt.figure(figsize=(20*mm, 16*mm), tight_layout=True)
nrows = 2
nvols = 3
 
# rewrading left
fig.add_subplot(nrows, nvols, 1)
ax = sns.barplot(
    data = leftCorrect, x='group', y='leftCorrect', hue='Condition',  
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.1,
    alpha=0.5,
    errorbar="sd", legend=True)
gfg = sns.stripplot(
    data=leftCorrect, x='group', y='leftCorrect', hue='Condition',
     dodge=True, alpha=0.6, ax=ax, legend=False
)

# for legend text
plt.setp(gfg.get_legend().get_texts(), fontsize='8')  
plt.setp(gfg.get_legend().get_title(), fontsize='8')  
plt.title('')
plt.xlabel('')
plt.ylabel('Correct left', fontsize='10')
plt.axhline(.5, color='black' , linestyle='--')
plt.ylim(0, .9)

# rewarding push
fig.add_subplot(nrows, nvols, 2)
ax = sns.barplot(
    data = pushCorrect, x='group', y='pushCorrect', hue='Condition',  
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.1,
    alpha=0.5,
    errorbar="sd", legend=False)
sns.stripplot(
    data=pushCorrect, x='group', y='pushCorrect', hue='Condition', 
     dodge=True, alpha=0.6, ax=ax, legend=False
)

plt.title('')
plt.xlabel('')
plt.ylabel('Correct push', fontsize='10')
plt.axhline(.5, color='black' , linestyle='--')
plt.ylim(0, .9)
 
# rewrding yellow 
fig.add_subplot(nrows, nvols, 3)
ax = sns.barplot(
    data = yellowCorrect, x='group', y='yellowCorrect', hue='Condition',  
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.1,
    alpha=0.5,
    errorbar="sd", legend=False)
sns.stripplot(
    data=yellowCorrect, x='group', y='yellowCorrect', hue='Condition', 
     dodge=True, alpha=0.6, ax=ax, legend=False
)
#plt.text(x=0, y=.51, s='***')
#plt.text(x=1, y=.51, s='***')
#plt.text(x=2, y=.51, s='*')
plt.title('')
plt.xlabel('')
plt.ylabel('Correct yellow', fontsize='10')
plt.axhline(.5, color='black' , linestyle='--')
plt.ylim(0, .9)


"""Choice biase based on the task"""
# read collected data across data
behAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/Amin/BehData/AllBehData/behAll.csv')
# gtoup label 1,2 3 are PD-OFF, HC and PF-ON respectively
behAll['group'] = behAll['group'].replace([1,2,3], ['PD-OFF', 'HC', 'PD-ON'])
# gtoup label 1,2 3 are PD-OFF, HC and PF-ON respectively
behAll['Condition'] = behAll['block'].replace(['Acyt', 'Stim'], ['Act', 'Clr'])

print(behAll.columns)

# Find chosen amount for each trial
chosenAmount = behAll['leftChosen']*behAll['winAmtLeft'] + (1-behAll['leftChosen'])*behAll['winAmtRight'] 
# Calculate the probability of high amount is chosed or lower amount
behAll['chosenHighWinAmt'] = chosenAmount>=50
# Calculuate reward left option
leftCorrect = behAll['leftCanBePushed                ']*behAll.pushCorrect + (1-behAll['leftCanBePushed                '])*(1-behAll.pushCorrect)
behAll['leftCorrect'] = leftCorrect

# the proportion of rewarded left, push and chosen yellow 
leftChosen = behAll[behAll['chosenHighWinAmt']==1].groupby(['group', 'Condition', 'sub_ID'], as_index=False)['leftChosen'].mean()
pushed = behAll[behAll['chosenHighWinAmt']==1].groupby(['group', 'Condition', 'sub_ID'], as_index=False)['pushed'].mean()
yellowChosen = behAll[behAll['chosenHighWinAmt']==1].groupby(['group', 'Condition', 'sub_ID'], as_index=False)['yellowChosen'].mean()

 
# rewrading left
fig.add_subplot(nrows, nvols, 4)
ax = sns.barplot(
    data = leftChosen, x='group', y='leftChosen', hue='Condition', 
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.1,
    alpha=0.5,
    errorbar="sd", legend=False)
sns.stripplot(
    data=leftChosen, x='group', y='leftChosen', hue='Condition',
     dodge=True, alpha=0.6, ax=ax, legend=False
)

plt.title('')
plt.xlabel('', fontsize='10')
plt.ylabel('Left reponse', fontsize='10')
plt.axhline(.5, color='black' , linestyle='--')
plt.ylim(0, .9)

# rewarding push
fig.add_subplot(nrows, nvols, 5)
ax = sns.barplot(
    data = pushed, x='group', y='pushed', hue='Condition',
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.1,
    alpha=0.5,
    errorbar="sd", legend=False)
sns.stripplot(
    data=pushed, x='group', y='pushed', hue='Condition', 
     dodge=True, alpha=0.6, ax=ax, legend=False
)
plt.title('')
plt.xlabel('', fontsize='10')
plt.ylabel('Push reponse', fontsize='10')
plt.axhline(.5, color='black' , linestyle='--')
plt.ylim(0, .9)
 
# rewrding yellow 
fig.add_subplot(nrows, nvols, 6)
ax = sns.barplot(
    data = yellowChosen, x='group', y='yellowChosen', hue='Condition',  
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.1,
    alpha=0.5,
    errorbar="sd", legend=False)
sns.stripplot(
    data=yellowChosen, x='group', y='yellowChosen', hue='Condition', 
     dodge=True, alpha=0.6, ax=ax, legend=False
)
plt.title('')
plt.xlabel('', fontsize='10')
plt.ylabel('Yellow reponse', fontsize='10')
plt.axhline(.5, color='black' , linestyle='--')
plt.ylim(0, .9)

# group title
fig.supxlabel('Group label')
fig.suptitle('Exsisting bias for each feature across group and condition', fontsize='12')

# save figure
plt.savefig('../../Figures/rewarded_options_bias_task.png', dpi=300)
plt.show()
