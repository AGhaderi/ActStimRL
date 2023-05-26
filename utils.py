import numpy as np



def simulate_rl(task_frame, alpha_A, alpha_C, weight, ini_learning=0):
    """Simulates a single participant behavior for Action and Stimulus Value Learning 
    according to a RL model with the weightening parameter,

    where the learning component is the Q learning
    (delta learning rule) and the choice rule is the softmax.

    This function is to simulate data for, for example, parameter recovery.
    Simulates data for one participant.

    Note
    ----
    The number of options can be actaully higher than 2,
    but only 2 options (one correct, one incorrect) are presented
    in each trial.
    It is important that "trial_block" is set to 1 at the beginning
    of each learning session (when the Q values at resetted)
    and that the "block_label" is set to 1 at the beginning of the
    experiment for each participants.
    There is no special requirement for the participant number.

    Parameters
    ----------

    task_design : DataFrame
        `pandas.DataFrame`, with n_trials_block*n_blocks rows.
        Columns contain:
        "f_cor", "f_inc", "trial_type", "cor_option", "inc_option",
        "trial_block", "block_label", "participant".

    gen_alpha : float or list of floats
        The generating learning rate.
        It should be a value between 0 (no updating) and 1 (full updating).
        If a list of 2 values is provided then 2 separate learning rates
        for positive and negative prediction error are used.

    gen_sensitivity : float
        The generating sensitivity parameter for the soft_max choice rule.
        It should be a value higher than 0
        (the higher, the more sensitivity to value differences).

    initial_value_learning : float
        The initial value for Q learning.

    Returns
    -------

    data : DataFrame
        `pandas.DataFrame`, that is the task_design, plus:
        'Q_cor', 'Q_inc', 'alpha', 'sensitivity',
        'p_cor', and 'accuracy'.

    """
    data = task_design.copy()

    if (type(gen_alpha) == float) | (type(gen_alpha) == int):
        data['alpha'] = gen_alpha
        data = pd.concat([data, _simulate_delta_rule_2A(task_design=task_design,
                                                                   alpha=gen_alpha,
                                                                   initial_value_learning=initial_value_learning)],
                         axis=1)

    elif type(gen_alpha) is list:
        if len(gen_alpha) == 2:
            data['alpha_pos'] = gen_alpha[0]
            data['alpha_neg'] = gen_alpha[1]
            data = pd.concat([data, _simulate_delta_rule_2A(task_design=task_design,
                                                                       alpha=None,
                                                                       initial_value_learning=initial_value_learning,
                                                                       alpha_pos=gen_alpha[0],
                                                                       alpha_neg=gen_alpha[1])],
                             axis=1)

        elif len(gen_alpha) == 3:
            pass # implement here Stefano's learning rule
        else:
            raise ValueError("The gen_alpha list should be of either length 2 or 3.")
    else:
        raise TypeError("The gen_alpha should be either a list or a float/int.")

    data['sensitivity'] = gen_sensitivity
    data['p_cor'] = data.apply(_soft_max_2A, axis=1)
    data['accuracy'] = stats.bernoulli.rvs(data['p_cor'].values) # simulate choices

    data = data.set_index(['participant', 'block_label', 'trial_block'])
    return data
