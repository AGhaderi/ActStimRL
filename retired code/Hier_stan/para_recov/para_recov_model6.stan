/* Model 5 assumes beta distribution in group level

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
    real<lower=0, upper=1> p_push_init;         // Initial value of reward probability for pushed responce
    real<lower=0, upper=1> p_yell_init;         // Initial value of reward probability for Color responce
}
parameters {
    /* Hierarchical mu parameter*/                               
    real<lower=0>  hier_alphaAct_a;    // Alpha Hierarchical Learning rate for Action Learning Value
    real<lower=0>  hier_alphaAct_b;    // Beta Hierarchical Learning rate for Action Learning Value
    real<lower=0>  hier_alphaClr_a;    // Alpha Hierarchical Learning rate for Color Learning Value
    real<lower=0>  hier_alphaClr_b;    // Beta Hierarchical Learning rate for Color Learning Value
    real<lower=0>  hier_weightAct_a;   // Alpha Hierarchical Wieghtening of Action Learning Value against to Color Learnig Value
    real<lower=0>  hier_weightAct_b;   // Beta Hierarchical Wieghtening of Action Learning Value against to Color Learnig Value
    real<lower=0> hier_sensitivity_mu;    // Mean Hierarchical snesitivity, With a higher sensitivity value θ, choices are more sensitive to value differences
    real<lower=0> hier_sensitivity_sd;    // Between-participant variability sensitivity

    /* participant-level main paameter*/
    array[nParts] real<lower=0, upper=1> alphaAct;   // Learning rate for Action Learning Value
    array[nParts] real<lower=0, upper=1> alphaClr;   // Learning rate for Color Learning Value
    array[nParts] real<lower=0, upper=1> weightAct;  // Wieghtening of Action Learning Value against to Color Learnig Value
    array[nParts] real<lower=0> sensitivity;         // With a higher sensitivity value θ, choices are more sensitive to value differences

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
    vector[N] sm_act;  //  The soft-max function for each trial for Acion values
    vector[N] sm_clr;  //  The soft-max function for each trial for color values

    // Calculating the probability of reward
   for (i in 1:N) {
        if (indicator[i]==1){
            p_push = p_push_init;
            p_pull = 1 - p_push_init;
            p_yell = p_yell_init;
            p_blue = 1 - p_yell_init;
        }
        
        // Action-based decision  
        // Calculating the Standard Expected Value
        EV_push = p_push*winAmtPushable[i];
        EV_pull = p_pull*winAmtPullable[i];
        // soft max function
        sm_act[i] = exp(sensitivity[participant[i]]*EV_push)/(exp(sensitivity[participant[i]]*EV_push) + exp(sensitivity[participant[i]]*EV_pull));

        // Color-based decision  
        // Calculating the Standard Expected Value
        EV_yell = p_yell*winAmtYellow[i];
        EV_blue = p_blue*winAmtBlue[i];
        // soft max function 
        sm_clr[i] = exp(sensitivity[participant[i]]*EV_yell)/(exp(sensitivity[participant[i]]*EV_yell) + exp(sensitivity[participant[i]]*EV_blue));  
   
        // RL rule update for computing prediction error and internal value expectation for the next trial based on the current reward output and interal value expectation
        if (pushed[i] == 1){
            p_push = p_push + alphaAct[participant[i]]*(rewarded[i] - p_push); 
            p_pull = 1 - p_push;
        }
        else{
            p_pull = p_pull + alphaAct[participant[i]]*(rewarded[i] - p_pull);
            p_push = 1 - p_pull;
        }    
        if (yellowChosen[i] == 1){
           p_yell = p_yell + alphaClr[participant[i]]*(rewarded[i] - p_yell);
           p_blue = 1 - p_yell;
        }    
        else{
           p_blue = p_blue + alphaClr[participant[i]]*(rewarded[i] - p_blue);
           p_yell = 1 - p_blue;           
        }
       
    }   
}
model { 
    /* Hierarchical mu parameter*/    
    hier_alphaAct_a ~ normal(0,3);
    hier_alphaAct_b ~ normal(0,3); 
    hier_alphaClr_a ~ normal(0,3); 
    hier_alphaClr_b ~ normal(0,3);
    hier_weightAct_a ~ normal(3,1); 
    hier_weightAct_b ~ normal(3,1);
    hier_sensitivity_mu ~ normal(0,1); 
    hier_sensitivity_sd ~ normal(0,1);
    
    /* participant-level main paameter*/
    for (p in 1:nParts) {
        alphaAct[p] ~ beta(hier_alphaAct_a, hier_alphaAct_b); 
        alphaClr[p] ~ beta(hier_alphaClr_a, hier_alphaClr_b);
        weightAct[p] ~ beta(hier_weightAct_a, hier_weightAct_b);
        sensitivity[p] ~ normal(hier_sensitivity_mu, hier_sensitivity_sd); 
    }

    /* RL likelihood */
    for (i in 1:N) { 
        // Relative contribution of Action Value Learning verus Color Value Learning
        target += log_sum_exp(log(weightAct[participant[i]]) + binomial_lpmf(pushed[i] | 1, sm_act[i]),
                              log1m(weightAct[participant[i]]) + binomial_lpmf(yellowChosen[i] | 1, sm_clr[i]));
    }
}
generated quantities { 
   vector[N] log_lik;  
    /*  RL Log density likelihood */
    for (i in 1:N) {
        log_lik[i] = log_sum_exp(log(weightAct[participant[i]]) + binomial_lpmf(pushed[i] | 1, sm_act[i]),
                              log1m(weightAct[participant[i]]) + binomial_lpmf(yellowChosen[i] | 1, sm_clr[i]));
        }
}