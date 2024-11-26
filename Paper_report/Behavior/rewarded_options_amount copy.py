"""Intrinsic tendency of reward probability toward left/ right, push/pull, yellow/blue across participant for each condition."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# read collected data across data
behAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/behAll.csv')
# gtoup label 1,2 3 are PD-OFF, HC and PF-ON respectively
behAll['group'] = behAll['group'].replace([1,2,3], ['PD-OFF', 'HC', 'PD-ON'])
# gtoup label 1,2 3 are PD-OFF, HC and PF-ON respectively
behAll['Condition'] = behAll['block'].replace(['Acyt', 'Stim'], ['Act', 'Clr'])

print(behAll.columns)

# object expected value
behAll['EV_pushed'] = behAll['winAmtPushable']* behAll['pushCorrect']
behAll['EV_yellow'] = behAll['winAmtYellow']* behAll['yellowChosen']

# sum of expected value for push and yellow chosen
EV_group = behAll.groupby(['Condition', 'sub_ID'], as_index=False)[['EV_pushed', 'EV_yellow']].mean()

# plot
mm = 1/2.54  # centimeters in inches
fig = plt.figure(figsize=(20*mm, 16*mm), tight_layout=True)
nrows = 1
nvols = 1

# rewrading left
fig.add_subplot(nrows, nvols, 1)


ax = sns.lmplot(behAll, x='EV_pushed', y='EV_yellow', hue='Condition')

plt.title('')
plt.xlabel('Expected Value', fontsize='10')
plt.ylabel('Expected Value', fontsize='10')
#plt.ylim(0, .9)

# group title
#fig.supxlabel('Group label')
#fig.suptitle('Exsisting bias for each feature across group and condition', fontsize='12')

# save figure
#plt.savefig('Figures/rewarded_options_bias_task.png', dpi=300)
plt.show()
