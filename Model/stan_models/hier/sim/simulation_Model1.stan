/* Model 2 assumes that there are meaningful effect across condition and group (HC and PD).

    This cond contains RL in addditon to weightening parameter to bels  both Action and Color values learning at the same time
    with Hierarchical level nanalysis.
    For both distinct options the push is coded 1 if selected and 0 if not selected. But for the first option, yellow chosen encoded 1 and blue is conded 0
    on the other hand, in the second option blue is encoded 1 and yellow is encoded 0. At the result, since the push matches to yellow in the first option
    and maches the blue in the second option, there is no necessary to change anything, we should just consider the push encoding.
*/ 


data {
    int<lower=1> N;        // Number of trial-level observations
    int<lower=1> nParts;   // Number of participants
    array[N] int<lower=0, upper=1> pushed;             // 1 if pushed and 0 if pulled 
    array[N] int<lower=0, upper=1> yellowChosen;       // 1 if yellow color is chosen and 0 if yellow color is not chosen 
    array[N] real<lower=0, upper=100> winAmtPushable;  // The amount of values feedback when pushing is correct response
    array[N] real<lower=0, upper=100> winAmtPullable;  // The amount of values feedback when pulling is correct response
    array[N] real<lower=0, upper=100> winAmtYellow;    // The amount of values feedback when yellow chosen is correct response 
    array[N] real<lower=0, upper=100> winAmtBlue;      // The amount of values feedback when blue chosen is correct response 
    array[N] int<lower=0, upper=1> rewarded;           // 1 for rewarding and 0 for punishment
    array[N] int<lower=1> participant;                 // participant index for each trial
    array[N] int<lower=1> indicator;                   // indicator of the first trial of each participant, the first is denoted 1 otherwise 0
    int<lower=1> nGrps;    // Number of groups, Helthy Control and Parkinson Disease
    int<lower=1> nConds;   // Number of conditions, Action and Color value learning
    array[N] int<lower=1, upper=2> group;       // 1 indicate Hleathy Control and 2 indicates Parkinson Disease
    array[N] int<lower=1, upper=2> condition;   // 1 indecates Action value learning and 2 indicates Color value learning
    real<lower=0, upper=1> p_push_init;         // Initial value of reward probability for pushed responce
    real<lower=0, upper=1> p_yell_init;         // Initial value of reward probability for Color responce
}
parameters {
    /* Hierarchical mu parameter*/                               
    array[nGrps, nConds] real hier_alphaAct_mu;    // Mean Hierarchical Learning rate for Action Learning Value
    array[nGrps, nConds] real hier_alphaClr_mu;    // Mean Hierarchical Learning rate for Color Learning Value
    array[nGrps, nConds] real hier_weightAct_mu;   // Mean Hierarchical Wieghtening of Action Learning Value against to Color Learnig Value
    array[nGrps, nConds] real hier_sensitivity_mu;    // Mean Hierarchical snesitivity, With a higher sensitivity value θ, choices are more sensitive to value differences

    /* Hierarchical sd parameter*/                               
    real<lower=0> hier_alphaAct_sd;      // Between-participant variability Learning rate for Action Learning Value
    real<lower=0> hier_alphaClr_sd;      // Between-participant variability Learning rate for Color Learning Value
    real<lower=0> hier_weightAct_sd;     // Between-participant variability Wieghtening of Action Learning Value against to Color Learnig Value
    real<lower=0> hier_sensitivity_sd;   // Between-participant variability sensitivity

    /* participant-level main paameter*/
    array[nParts, nGrps, nConds] real z_alphaAct;   // Learning rate for Action Learning Value
    array[nParts, nGrps, nConds] real z_alphaClr;   // Learning rate for Color Learning Value
    array[nParts, nGrps, nConds] real z_weightAct;  // Wieghtening of Action Learning Value against to Color Learnig Value
    array[nParts, nGrps, nConds] real z_sensitivity;         // With a higher sensitivity value θ, choices are more sensitive to value differences

}
transformed parameters {
    /* probability of each features and their combination */
    real p_push;   // Probability of reward for pushing responce
    real p_pull;   // Probability of reward for pulling responce
    real p_yell;   // Probability of reward for yrllow responce
    real p_blue;   // Probability of reward for blue responce
    real EV_push;  // Standard Expected Value of push action
    real EV_pull;  // Standard Expected Value of pull action
    real EV_yell;  // Standard Expected Value of yellow action
    real EV_blue;  // Standard Expected Value of blue action
    real EV_push_yell;      // Weighting two strategies between push action and yellow color values learning
    real EV_push_blue;      // Weighting two strategies between push action and blue color values learning
    real EV_pull_yell;      // Weighting two strategies between pull action and yellow color values learning
    real EV_pull_blue;      // Weighting two strategies between pull action and blue color values learning
    vector[N] soft_max_EV;  //  The soft-max function for each trial, trial-by-trial probability
   
    /* Transfer individual parameters */
    array[nParts, nGrps, nConds] real<lower=0, upper=1> transfer_alphaAct;   // Learning rate for Action Learning Value
    array[nParts, nGrps, nConds] real<lower=0, upper=1> transfer_alphaClr;   // Learning rate for Color Learning Value
    array[nParts, nGrps, nConds] real<lower=0, upper=1> transfer_weightAct;  // Wieghtening of Action Learning Value against to Color Learnig Value
    array[nParts, nGrps, nConds] real<lower=0> transfer_sensitivity;         // With a higher sensitivity value θ, choices are more sensitive to value differences
    
    /* Transfer Hierarchical parameters just for output*/
    array[nGrps, nConds] real<lower=0, upper=1> transfer_hier_alphaAct_mu;   // Hierarchical Learning rate for Action Learning Value
    array[nGrps, nConds] real<lower=0, upper=1> transfer_hier_alphaClr_mu;   // Hierarchical Learning rate for Color Learning Value
    array[nGrps, nConds] real<lower=0, upper=1> transfer_hier_weightAct_mu;  // Hierarchical Wieghtening of Action Learning Value against to Color Learnig Value
    array[nGrps, nConds] real<lower=0> transfer_hier_sensitivity_mu;         // Hierarchical snesitivity, With a higher sensitivity value θ, choices are more sensitive to value differences

	transfer_hier_alphaAct_mu = Phi(hier_alphaAct_mu);				// for the output
    transfer_hier_alphaClr_mu = Phi(hier_alphaClr_mu);
    transfer_hier_weightAct_mu = Phi(hier_weightAct_mu);
    for (m in 1:nGrps){
        for (c in 1:nConds){
	        transfer_hier_sensitivity_mu[m, c] = log(1 + exp(hier_sensitivity_mu[m, c]));
        }
    }

    for (p in 1:nParts) {
        for (m in 1:nGrps){
            for (c in 1:nConds){
                transfer_alphaAct[p, m, c] = Phi(hier_alphaAct_mu[m, c] + z_alphaAct[p, m, c]*hier_alphaAct_sd);
                transfer_alphaClr[p, m, c] = Phi(hier_alphaClr_mu[m, c] + z_alphaClr[p, m, c]*hier_alphaClr_sd);
                transfer_weightAct[p, m, c] = Phi(hier_weightAct_mu[m, c] + z_weightAct[p, m, c]*hier_weightAct_sd);
                transfer_sensitivity[p, m, c] = log(1 + exp(hier_sensitivity_mu[m, c] + z_sensitivity[p, m, c]*hier_sensitivity_sd));
            }
        } 
    }

    // Calculating the probability of reward
   for (i in 1:N) {
        if (indicator[i]==1){
            p_push = p_push_init;
            p_pull = 1 - p_push_init;
            p_yell = p_yell_init;
            p_blue = 1 - p_yell_init;
        }
        // Calculating the Standard Expected Value
        EV_push = p_push*winAmtPushable[i];
        EV_pull = p_pull*winAmtPullable[i];
        EV_yell = p_yell*winAmtYellow[i];
        EV_blue = p_blue*winAmtBlue[i];
       
        // Relative contribution of Action Value Learning verus Color Value Learning
        EV_push_yell = transfer_weightAct[participant[i], group[i], condition[i]]*EV_push + (1 - transfer_weightAct[participant[i], group[i], condition[i]])*EV_yell;
        EV_push_blue = transfer_weightAct[participant[i], group[i], condition[i]]*EV_push + (1 - transfer_weightAct[participant[i], group[i], condition[i]])*EV_blue;
        EV_pull_yell = transfer_weightAct[participant[i], group[i], condition[i]]*EV_pull + (1 - transfer_weightAct[participant[i], group[i], condition[i]])*EV_yell;
        EV_pull_blue = transfer_weightAct[participant[i], group[i], condition[i]]*EV_pull + (1 - transfer_weightAct[participant[i], group[i], condition[i]])*EV_blue;
       
        /* Calculating the soft-max function over weightening Action and Color conditions*/ 
        // pushed and yellow vs pulled and blue
        if ((pushed[i] == 1 && yellowChosen[i] == 1) || (pushed[i] == 0 && yellowChosen[i] == 0))
            soft_max_EV[i] = exp(transfer_sensitivity[participant[i], group[i], condition[i]]*EV_push_yell)/(exp(transfer_sensitivity[participant[i], group[i], condition[i]]*EV_push_yell) + exp(transfer_sensitivity[participant[i], group[i], condition[i]]*EV_pull_blue));

        // pushed and blue vs pulled and yellow
        if ((pushed[i] == 1 && yellowChosen[i] == 0) || (pushed[i] == 0 && yellowChosen[i] == 1))
            soft_max_EV[i] = exp(transfer_sensitivity[participant[i], group[i], condition[i]]*EV_push_blue)/(exp(transfer_sensitivity[participant[i], group[i], condition[i]]*EV_push_blue) + exp(transfer_sensitivity[participant[i], group[i], condition[i]]*EV_pull_yell));  
          
        // RL rule update for computing prediction error and internal value expectation for the next trial based on the current reward output and interal value expectation
        if (pushed[i] == 1){
            p_push = p_push + transfer_alphaAct[participant[i], group[i], condition[i]]*(rewarded[i] - p_push); 
            p_pull = 1 - p_push;
        }
        else{
            p_pull = p_pull + transfer_alphaAct[participant[i], group[i], condition[i]]*(rewarded[i] - p_pull);
            p_push = 1 - p_pull;
        }    
        if (yellowChosen[i] == 1){
           p_yell = p_yell + transfer_alphaClr[participant[i], group[i], condition[i]]*(rewarded[i] - p_yell);
           p_blue = 1 - p_yell;
        }    
        else{
           p_blue = p_blue + transfer_alphaClr[participant[i], group[i], condition[i]]*(rewarded[i] - p_blue);
           p_yell = 1 - p_blue;           
        }
    }   
}
model { 
    /* Hierarchical mu parameter*/    
    for (m in 1:nGrps){
        for (c in 1:nConds){
            hier_alphaAct_mu[m, c] ~ normal(0,1);
            hier_alphaClr_mu[m, c] ~ normal(0,1); 
            hier_sensitivity_mu[m, c] ~ normal(1,5); 
            hier_weightAct_mu[m, c] ~ normal(0,1);
        }
    } 
 
    /* Hierarchical sd parameter*/
    hier_alphaAct_sd ~ normal(0,1) T[0,];  
    hier_alphaClr_sd ~ normal(0,1) T[0,];
    hier_weightAct_sd ~ normal(0,1) T[0,]; 
    hier_sensitivity_sd ~ normal(0,1) T[0,];
    
    /* participant-level main paameter*/
    for (p in 1:nParts) {
        for (m in 1:nGrps){
            for (c in 1:nConds){
                z_alphaAct[p, m, c] ~ normal(0,1); 
                z_alphaClr[p, m, c] ~ normal(0,1);
                z_weightAct[p, m, c] ~ normal(0,1);
                z_sensitivity[p, m, c] ~ normal(0,1); 
            }   
        } 
    }

    /* RL likelihood */
    for (i in 1:N) { 
        pushed[i] ~ bernoulli(soft_max_EV[i]);
        }
}
generated quantities { 
   vector[N] log_lik;  
    /*  RL Log density likelihood */
    for (i in 1:N) {
        log_lik[i] = bernoulli_lpmf(pushed[i] | soft_max_EV[i]);
        }
}