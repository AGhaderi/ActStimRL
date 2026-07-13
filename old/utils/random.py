import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
from .config import (
    PROJECT_RAW_BEH_ALL_FILE, PROJECT_NoNAN_BEH_ALL_FILE, FIGURES_DIR, OUTPUT_CSV_DIR
    )

def simStrategyBehavior(BehDir = PROJECT_RAW_BEH_ALL_FILE, saveFile= FIGURES_DIR):
    """
    Simulate multiple agent strategies and compare them with observed behavioral data.

    This function encompasses:
    - Calculation of agent choices according to different strategies (higher/lower amount, win-stay/lose-shift,
      random, probability-based, and oracle).
        - Higher/Larger Amount Agent:
            Always chooses the option with the higher reward amount.
        - Lower/Smaller Amount Agent
            Always chooses the option with the lower reward amount.
        - Win-Stay / Lose-Shift Agent
            Follows a simple reinforcement heuristic:
            Win → stay: If previous choice was rewarded, repeat it.
            Lose → shift: If previous choice was not rewarded, switch to the other option.
        - Random Agent
            Chooses push/pull randomly with equal probability (50/50).
        - Probability-Based Agent
            Chooses the option with the ideal higher probability of being rewarded. 
            This agent is used to see of the probality of reward leads to higher amout
        - Oracle Agent
            Chooses the option with the highest expected value (amount * probability).
    - Aggregation of rewarded amount is used data for subjects in each group (HC. PD-OFF and PD-ON)
    - Saving the final plot as a PDF.

    Parameters
    ----------
    BehDir : str, optional
        File path to the behavioral data CSV file. Default is `RAW_BEH_ALL_FILE`.
    saveFile : str, optional
        Directory or file path to save the generated plot (PDF). Default is `FIGURES_DIR`.

    Returns
    -------
    None
        The function saves the plot to the specified path and does not return any objects.
    """
 
    # Load behavioral data collected across all participants
    behAll = pd.read_csv(BehDir)

    # Replace group labels: 1, 2, 3 → PD-OFF, HC, PD-ON
    behAll['group'] = behAll['group'].replace([1, 2, 3], ['PD-OFF', 'HC', 'PD-ON'])

    # Replace block labels
    behAll['block'] = behAll['block'].replace(['Act', 'Stim'], ['Act', 'Clr'])

    # Rename columns with inconsistent spacing
    behAll = behAll.rename(columns={'wonAmount                ': 'wonAmount',
                                    'leftCanBePushed                ': 'leftCanBePushed'})

    # Compute whether the left option was the rewarded option
    leftCorrect = behAll['leftCanBePushed'] * behAll.pushCorrect + (1 - behAll['leftCanBePushed']) * (1 - behAll.pushCorrect)
    behAll['leftCorrect'] = leftCorrect


    ########################################### 1. Choosing the option with the higher amount
    leftCorrect_higher_amt = behAll['leftCorrect'] * (behAll['winAmtLeft'] >= 50) * behAll['winAmtLeft']
    rightCorrect_higher_amt = (1 - behAll['leftCorrect']) * (behAll['winAmtBlue'] >= 50) * behAll['winAmtBlue']
    agent_higher_amount = leftCorrect_higher_amt + rightCorrect_higher_amt
    behAll['agent_higher_amount'] = agent_higher_amount


    ############################################ 2. Choosing the option with the lower amount
    leftCorrect_lower_amt = behAll['leftCorrect'] * (behAll['winAmtLeft'] < 50) * behAll['winAmtLeft']
    rightCorrect_lower_amt = (1 - behAll['leftCorrect']) * (behAll['winAmtBlue'] < 50) * behAll['winAmtBlue']
    agent_lower_amount = leftCorrect_lower_amt + rightCorrect_lower_amt
    behAll['agent_lower_amount'] = agent_lower_amount


    ########################################### 3. Win-stay / lose-shift agent
    """This agent follows a win-stay/lose-shift rule for push/pull in the Action condition 
    and yellow/blue in the Color condition."""

    choiceWinLose = np.zeros(behAll.shape[0])
    # 1 = push/yellow, 0 = pull/blue

    for i in range(behAll.shape[0] - 1):
        # Identify the rewarded option according to the block type
        if behAll['block'][i] == 'Act':
            correct = 'pushCorrect'
        elif behAll['block'][i] == 'Clr':
            correct = 'yellowCorrect'

        # Apply win-stay / lose-shift logic
        if (choiceWinLose[i] == 1) and (choiceWinLose[i] == behAll[correct][i]):
            choiceWinLose[i + 1] = 1   # win → stay (push or yellow)
        elif (choiceWinLose[i] == 0) and (choiceWinLose[i] == behAll[correct][i]):
            choiceWinLose[i + 1] = 0   # win → stay (pull or blue)
        elif (choiceWinLose[i] == 1) and (choiceWinLose[i] != behAll[correct][i]):
            choiceWinLose[i + 1] = 0   # lose → shift
        elif (choiceWinLose[i] == 0) and (choiceWinLose[i] != behAll[correct][i]):
            choiceWinLose[i + 1] = 1   # lose → shift

    # Compute rewarded amounts for win-stay/lose-shift agent
    behAll['agent_winStay_shiftLose'] = (
        (behAll['block'] == 'Act') *
        (choiceWinLose * behAll['pushCorrect'] * behAll['winAmtPushable'] +
        (1 - choiceWinLose) * (1 - behAll['pushCorrect']) * behAll['winAmtPullable'])
        +
        (behAll['block'] == 'Clr') *
        (choiceWinLose * behAll['yellowCorrect'] * behAll['winAmtYellow'] +
        (1 - choiceWinLose) * (1 - behAll['yellowCorrect']) * behAll['winAmtBlue'])
    )


    ########################################### 4. Random agent: chooses push or pull randomly
    # Generate random choices: 1 = push, 0 = pull
    rand = np.random.binomial(1, .5, size=behAll.shape[0])

    # Compute rewarded amounts for random agent
    behAll['agent_random'] = (
        behAll['pushCorrect'] * rand * behAll['winAmtPushable'] +
        (1 - behAll['pushCorrect']) * (1 - rand) * behAll['winAmtPullable']
    )


    ########################################### 5. Agent choosing the option with higher reward probability
    # Placeholder for probability-based choices
    behAll['prob_based_choices'] = np.nan

    # Subject list
    subList = behAll['sub_ID'].unique()

    for subName in subList:
        for sess in [1, 2]:
            for run in [1, 2]:
                for cond in ['Clr', 'Act']:

                    # Select data for specific session/run/condition
                    actData = behAll[(behAll['session'] == sess) & (behAll['run'] == run) &
                                    (behAll['block'] == cond) & (behAll['sub_ID'] == subName)]

                    # Phase-wise processing
                    phases = actData['phase'].unique()
                    for phase in phases:
                        actDataPhase = actData[actData['phase'] == phase]

                        # Compute reward probability
                        if cond == 'Act':
                            propPhase = actDataPhase['pushCorrect'].mean()
                        if cond == 'Clr':
                            propPhase = actDataPhase['yellowCorrect'].mean()

                        # Choose the option with higher reward probability
                        behAll.loc[(behAll['session'] == sess) & (behAll['run'] == run) &
                                (behAll['block'] == cond) & (behAll['sub_ID'] == subName) &
                                (behAll['phase'] == phase), 'prob_based_choices'] = round(propPhase)

    # Reward amount for probability-based agent
    behAll['agent_hgiher_probability'] = (
        (behAll['block'] == 'Act') *
        (behAll['prob_based_choices'] * behAll['pushCorrect'] * behAll['winAmtPushable'] +
        (1 - behAll['prob_based_choices']) * (1 - behAll['pushCorrect']) * behAll['winAmtPullable'])
        +
        (behAll['block'] == 'Clr') *
        (behAll['prob_based_choices'] * behAll['yellowCorrect'] * behAll['winAmtYellow'] +
        (1 - behAll['prob_based_choices']) * (1 - behAll['yellowCorrect']) * behAll['winAmtBlue'])
    )


    ########################################### 6. Oracle agent: chooses the option with higher expected value
    behAll['agent_oracle'] = np.nan

    subList = behAll['sub_ID'].unique()

    for subName in subList:
        for sess in [1, 2]:
            for run in [1, 2]:
                for cond in ['Clr', 'Act']:

                    actData = behAll[(behAll['session'] == sess) & (behAll['run'] == run) &
                                    (behAll['block'] == cond) & (behAll['sub_ID'] == subName)]

                    phases = actData['phase'].unique()

                    for phase in phases:
                        actDataPhase = actData[actData['phase'] == phase]

                        if cond == 'Act':
                            # Compute expected values for push/pull
                            propPhase = actDataPhase['pushCorrect'].mean()
                            EV_bool = propPhase * actDataPhase['winAmtPushable'].to_numpy() > \
                                    (1 - propPhase) * actDataPhase['winAmtPullable'].to_numpy()

                            agent_oracle = (
                                EV_bool * actDataPhase['pushCorrect'] * actDataPhase['winAmtPushable'] +
                                (1 - EV_bool) * (1 - actDataPhase['pushCorrect']) * actDataPhase['winAmtPullable']
                            )
                            behAll.loc[(behAll['session'] == sess) & (behAll['run'] == run) &
                                    (behAll['block'] == cond) & (behAll['sub_ID'] == subName) &
                                    (behAll['phase'] == phase), 'agent_oracle'] = agent_oracle

                        elif cond == 'Clr':
                            # Compute expected values for yellow/blue
                            propPhase = actDataPhase['yellowCorrect'].mean()
                            EV_bool = propPhase * actDataPhase['winAmtYellow'].to_numpy() > \
                                    (1 - propPhase) * actDataPhase['winAmtBlue'].to_numpy()

                            agent_oracle = (
                                EV_bool * actDataPhase['yellowCorrect'] * actDataPhase['winAmtYellow'] +
                                (1 - EV_bool) * (1 - actDataPhase['yellowCorrect']) * actDataPhase['winAmtBlue']
                            )
                            behAll.loc[(behAll['session'] == sess) & (behAll['run'] == run) &
                                    (behAll['block'] == cond) & (behAll['sub_ID'] == subName) &
                                    (behAll['phase'] == phase), 'agent_oracle'] = agent_oracle



    # Check out if it does not exist
    if not os.path.isdir(f'{OUTPUT_CSV_DIR}/'):
        os.makedirs(f'{OUTPUT_CSV_DIR}/') 
    # save all synthetic agent and observation together
    behAll.to_csv(f'{OUTPUT_CSV_DIR}/simStrategyBehavior.csv', index=False)

    ############################################################# Plotting
    cm = 1/2.54  # centimeters in inches
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(21*cm, 12*cm))

    # Average total reward across subjects for observed data
    behAll_obs = behAll.groupby(['block', 'group', 'sub_ID'], as_index=False)['wonAmount'].sum()
    behAll_obs['wonAmount'] = behAll_obs['wonAmount'].astype(float)
    behAll_obs.loc[behAll_obs['group'] == 'HC', 'wonAmount'] /= 2

    behAll_obs_act_HC = behAll_obs[(behAll_obs['block'] == 'Act') & (behAll_obs['group'] == 'HC')]
    behAll_obs_act_PD_OFF = behAll_obs[(behAll_obs['block'] == 'Act') & (behAll_obs['group'] == 'PD-OFF')]
    behAll_obs_act_PD_ON = behAll_obs[(behAll_obs['block'] == 'Act') & (behAll_obs['group'] == 'PD-ON')]
    behAll_obs_clr_HC = behAll_obs[(behAll_obs['block'] == 'Clr') & (behAll_obs['group'] == 'HC')]
    behAll_obs_clr_PD_OFF = behAll_obs[(behAll_obs['block'] == 'Clr') & (behAll_obs['group'] == 'PD-OFF')]
    behAll_obs_clr_PD_ON = behAll_obs[(behAll_obs['block'] == 'Clr') & (behAll_obs['group'] == 'PD-ON')]

    # Average total reward across subjects for agents
    behAll_agent = behAll.groupby(['block', 'sub_ID'], as_index=False)[
        ['agent_higher_amount', 'agent_lower_amount', 'agent_winStay_shiftLose',
        'agent_random', 'agent_hgiher_probability', 'agent_oracle']].sum()

    behAll_agent_action = behAll_agent[behAll_agent['block'] == 'Act']
    behAll_agent_color = behAll_agent[behAll_agent['block'] == 'Clr']

    # X positions for agents
    agent_positions = np.arange(6) * 2
    offset = 0.3
    positions_agen_action = agent_positions - offset
    positions_agen_color = agent_positions + offset

    # Agent results for Action block
    means_agent_action = behAll_agent_action[['agent_lower_amount', 'agent_winStay_shiftLose',
                                              'agent_higher_amount', 'agent_oracle',
                                              'agent_hgiher_probability', 'agent_random']].mean(axis=0) / 2
    se_agent_action = behAll_agent_action[['agent_lower_amount', 'agent_winStay_shiftLose',
                                           'agent_higher_amount', 'agent_oracle',
                                           'agent_hgiher_probability', 'agent_random']].std(axis=0)

    axs.bar(positions_agen_action, means_agent_action, yerr=se_agent_action,
            capsize=5, width=.5, color='red', edgecolor='black')

    # Agent results for Color block
    means_agent_color = behAll_agent_color[['agent_lower_amount', 'agent_winStay_shiftLose',
                                            'agent_higher_amount', 'agent_oracle',
                                            'agent_hgiher_probability', 'agent_random']].mean(axis=0) / 2
    se_agent_color = behAll_agent_color[['agent_lower_amount', 'agent_winStay_shiftLose',
                                        'agent_higher_amount', 'agent_oracle',
                                        'agent_hgiher_probability', 'agent_random']].std(axis=0)

    axs.bar(positions_agen_color, means_agent_color, yerr=se_agent_color,
            capsize=5, width=.5, color='skyblue', edgecolor='black')

    # X positions for observed data
    obs_positions = np.array([15, 17, 19])
    positions_obs_action = obs_positions - offset
    positions_obs_color = obs_positions + offset

    # Observed Action results
    means_observation_action = [
        behAll_obs_act_HC['wonAmount'].mean(),
        behAll_obs_act_PD_OFF['wonAmount'].mean(),
        behAll_obs_act_PD_ON['wonAmount'].mean()
    ]
    se_observation_action = [
        behAll_obs_act_HC['wonAmount'].std(),
        behAll_obs_act_PD_OFF['wonAmount'].std(),
        behAll_obs_act_PD_ON['wonAmount'].std()
    ]

    axs.bar(positions_obs_action, means_observation_action, yerr=se_observation_action,
            capsize=5, width=.5, color='red', edgecolor='black', label='Act')

    # Observed Color results
    means_observation_color = [
        behAll_obs_clr_HC['wonAmount'].mean(),
        behAll_obs_clr_PD_OFF['wonAmount'].mean(),
        behAll_obs_clr_PD_ON['wonAmount'].mean()
    ]
    se_observation_color = [
        behAll_obs_clr_HC['wonAmount'].std(),
        behAll_obs_clr_PD_OFF['wonAmount'].std(),
        behAll_obs_clr_PD_ON['wonAmount'].std()
    ]

    axs.bar(positions_obs_color, means_observation_color, yerr=se_observation_color,
            capsize=5, width=.5, color='skyblue', edgecolor='black', label='Clr')

    # Axis labels and formatting
    axs.set_xticks(
        np.r_[agent_positions, obs_positions],
        ['Lower Amt', 'Win-Stay/Shift-Lose', 'Higher Amt', 'Oracle', 'Probability', 'Random',
        'HC', 'PD-OFF', 'PD-ON'],
        rotation=70
    )
    axs.set_ylim(0, 4500)
    axs.set_ylabel('Total amount', fontsize=12)
    axs.legend(title='Condition', loc='upper left')

    plt.tight_layout()

    # Check out if it does not exist
    if not os.path.isdir(f'{saveFile}/'):
        os.makedirs(f'{saveFile}/') 
    # Save figure
    fig.savefig(f'{saveFile}/simStrategyBehavior.pdf')
