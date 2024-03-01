""" Simulation study without RL model.
Agent chooses options with the higher probability of rewarding. """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Read synthetic agent related to higher winning porobability
higher_prob = pd.read_csv('Synthetic_agent/simulation_chosing_higher_probability.csv')
# Read synthetic agent related to higher amount
higher_amt = pd.read_csv('Synthetic_agent/simulation_chosing_higher_amount.csv')


# random selection of option between probability and amount
rand_prob_amt = np.random.binomial(1, .5, len(higher_prob))

pushed_agent = rand_prob_amt*higher_prob['pushed_agent'] + (1-rand_prob_amt)*higher_amt['pushed_agent']
yellowChosen_agent = rand_prob_amt*higher_prob['yellowChosen_agent'] + (1-rand_prob_amt)*higher_amt['yellowChosen_agent']
correctChoice_agent = rand_prob_amt*higher_prob['correctChoice_agent'] + (1-rand_prob_amt)*higher_amt['correctChoice_agent']

hgiher_prob_amt = higher_prob
hgiher_prob_amt['pushed_agent'] =pushed_agent
hgiher_prob_amt['yellowChosen_agent'] = yellowChosen_agent
hgiher_prob_amt['correctChoice_agent'] = correctChoice_agent

# Save datafram as csv
hgiher_prob_amt.to_csv('Synthetic_agent/simulation_higher_probability_higher_amount_50.csv', index=False)
 