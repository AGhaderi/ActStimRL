import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

# Read pooled data
behAll = pd.read_csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/data_BehModel/originalfMRIbehFiles/AllBehData/behAll.csv')
# Rename some columns label and entities
behAll = behAll.rename(columns={'wonAmount                ':'wonAmount'})

# Plot of won amount for each group
fig = plt.figure(figsize=(6, 4), tight_layout = True)
# accumulated won ammount for each participant, block and group label
behAll_totAmt = behAll.groupby(['group', 'block', 'sub_ID'], as_index=False)['wonAmount'].sum()
# We have two session1 and 2 for helathy control, so we should divide it by 2 to match with On and OFF medication
behAll_totAmt.loc[behAll_totAmt['group']==2,'wonAmount'] = behAll_totAmt[behAll_totAmt['group']==2]['wonAmount']/2
ax = sns.barplot(data = behAll_totAmt, x='group', y='wonAmount', hue='block', width=.5)
# Change the label of legend
new_title = 'Condition'
ax.legend_.set_title(new_title)
for t, l in zip(ax.legend_.texts, ['Act', 'Clr']):
    t.set_text(l)
plt.ylabel('Total amount')
plt.xlabel('Group label')

# Save figure
plt.savefig('../figures/total_amount.png', dpi=300)
plt.show()

# Mean of total amount across participant for each group label
print('Mean of total amount across participant: \n', behAll_totAmt.groupby(['group', 'block'])['wonAmount'].mean())
# STD of total amount across participant for each group label
print('STD of total amount across participant:\n', behAll_totAmt.groupby(['group', 'block'])['wonAmount'].std())
# Statistical test over total amount betwenn groups
print('Statistical test over total amount:\n', stats.ttest_rel(behAll_totAmt[behAll_totAmt['group']==3]['wonAmount'],
behAll_totAmt[behAll_totAmt['group']==1]['wonAmount']))