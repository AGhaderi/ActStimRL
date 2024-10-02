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
    int<lower=1> nMeds_nSes;                         // Number of medication_session level (OFF vs ON)
    int<lower=1> nConds;                        // Number of condition, Action and Color value learning
    array[N] int<lower=1, upper=2> medication_session;       // 1 indecates OFF medication_session and 2 indicates On medication_session
    array[N] int<lower=1, upper=2> condition;   // 1 indecates first codition (Action) and 2 indicates second condition (Color)
    real<lower=0, upper=1> p_push_init;         // Initial value of reward probability for pushed responce
    real<lower=0, upper=1> p_yell_init;         // Initial value of reward probability for Color responce
}
parameters {
    /* Hierarchical mu parameter*/                               
    array[nMeds_nSes, nConds] real hier_alphaAct_mu;    // Mean Hierarchical Learning rate for action Learning Value and medication_session effect
    array[nMeds_nSes, nConds] real hier_alphaClr_mu;    // Mean Hierarchical Learning rate for color Learning Value and medication_session effect
    array[nMeds_nSes, nConds] real hier_sensitivity_mu;    // Mean Hierarchical snesitivity, With a higher sensitivity value θ, choices are more sensitive to value differences

    /* Hierarchical sd parameter*/                               
    real<lower=0> hier_alpha_sd;      // Between-participant variability Learning rate for Learning Value
    real<lower=0> hier_sensitivity_sd;   // Between-participant variability sensitivity

    /* participant-level main paameter*/
    array[nParts, nMeds_nSes, nConds] real z_alphaAct;   // Learning rate for Action Learning Value
    array[nParts, nMeds_nSes, nConds] real z_alphaClr;   // Learning rate for Color Learning Value
    array[nParts, nMeds_nSes, nConds] real z_sensitivity;         // With a higher sensitivity value θ, choices are more sensitive to value differences

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
    vector[N] soft_max_Act;  //  The soft-max function for each trial for Action value learing
    vector[N] soft_max_Stim;  //  The soft-max function for each trial for Color value learning
   
    /* Transfer individual parameters */
    array[nParts, nMeds_nSes, nConds] real<lower=0, upper=1> transfer_alphaAct;   // Learning rate for Action Learning Value
    array[nParts, nMeds_nSes, nConds] real<lower=0, upper=1> transfer_alphaClr;   // Learning rate for Color Learning Value
    array[nParts, nMeds_nSes, nConds] real<lower=0> transfer_sensitivity;         // With a higher sensitivity value θ, choices are more sensitive to value differences
    
    /* Transfer Hierarchical parameters just for output*/
    array[nMeds_nSes, nConds] real<lower=0, upper=1> transfer_hier_alphaAct_mu;   // Hierarchical Learning rate for Action Learning Value
    array[nMeds_nSes, nConds] real<lower=0, upper=1> transfer_hier_alphaClr_mu;   // Hierarchical Learning rate for Color Learning Value
    array[nMeds_nSes, nConds] real<lower=0> transfer_hier_sensitivity_mu;         // Hierarchical snesitivity, With a higher sensitivity value θ, choices are more sensitive to value differences

	transfer_hier_alphaAct_mu = Phi(hier_alphaAct_mu);				// for the output
	transfer_hier_alphaClr_mu = Phi(hier_alphaClr_mu);				 
	for (g in 1:nMeds_nSes){
        for (c in 1:nConds){
            transfer_hier_sensitivity_mu[g, c] = log(1 + exp(hier_sensitivity_mu[g, c]));
        }
    }

    for (p in 1:nParts) {
        for (g in 1:nMeds_nSes){
            for (c in 1:nConds){
                transfer_alphaAct[p, g, c] = Phi(hier_alphaAct_mu[g, c] + z_alphaAct[p, g, c]*hier_alpha_sd);
                transfer_alphaClr[p, g, c] = Phi(hier_alphaClr_mu[g, c] + z_alphaClr[p, g, c]*hier_alpha_sd);
                transfer_sensitivity[p, g, c] = log(1 + exp(hier_sensitivity_mu[g, c] + z_sensitivity[p,g, c]*hier_sensitivity_sd));
            }
        }
    }

    // Calculating the probability of reward
   for (i in 1:N) {
        // Restart probability of variable for each environemnt and condition
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
       
        /* Calculating the soft-max function*/ 
        // pushed  vs pulled
        soft_max_Act[i] = exp(transfer_sensitivity[participant[i], medication_session[i], condition[i]]*EV_push)/(exp(transfer_sensitivity[participant[i], medication_session[i], condition[i]]*EV_push) + exp(transfer_sensitivity[participant[i], medication_session[i], condition[i]]*EV_pull));
        
        // yellow vs blue 
        soft_max_Stim[i] = exp(transfer_sensitivity[participant[i], medication_session[i], condition[i]]*EV_yell)/(exp(transfer_sensitivity[participant[i], medication_session[i], condition[i]]*EV_yell) + exp(transfer_sensitivity[participant[i], medication_session[i], condition[i]]*EV_blue));  
          
        // RL rule update for computing prediction error and internal value expectation for the next trial based on the current reward output and interal value expectation
        /*Action value learning*/
        if (pushed[i] == 1){
            p_push = p_push + transfer_alphaAct[participant[i], medication_session[i], condition[i]]*(rewarded[i] - p_push); 
        }
        else{
            p_pull = p_pull + transfer_alphaAct[participant[i], medication_session[i], condition[i]]*(rewarded[i] - p_pull);
        }   

        /*Color value learning*/
        if (yellowChosen[i] == 1){
           p_yell = p_yell + transfer_alphaClr[participant[i], medication_session[i], condition[i]]*(rewarded[i] - p_yell);
        }    
        else{
           p_blue = p_blue + transfer_alphaClr[participant[i], medication_session[i], condition[i]]*(rewarded[i] - p_blue);
        }
    }   
}
model { 
    /* Hierarchical mu parameter*/    
        for (g in 1:nMeds_nSes){
            for (c in 1:nConds){
                hier_alphaAct_mu[g,c] ~ normal(0,1);
                hier_alphaClr_mu[g,c] ~ normal(0,1);
                hier_sensitivity_mu[g,c] ~ normal(1,5); 
            }
        }

    /* Hierarchical sd parameter*/
    hier_alpha_sd ~ normal(0,.1);  
    hier_sensitivity_sd ~ normal(0,.1);
    
    /* participant-level main paameter*/
    for (p in 1:nParts) {
        for (g in 1:nMeds_nSes){
            for (c in 1:nConds){
                z_alphaAct[p, g, c] ~ normal(0,1);
                z_alphaClr[p, g, c] ~ normal(0,1);
                z_sensitivity[p, g, c] ~ normal(0,1); 
            }
        }
    }

    /* RL likelihood */
    for (i in 1:N) { 
        pushed[i] ~ bernoulli(soft_max_Act[i]);
        yellowChosen[i] ~ bernoulli(soft_max_Stim[i]);
        }
}
generated quantities { 
   vector[N] log_lik;  
    /*  RL Log density likelihood */
    for (i in 1:N) {
        log_lik[i] = bernoulli_lpmf(pushed[i] | soft_max_Act[i]) + bernoulli_lpmf(yellowChosen[i] | soft_max_Stim[i]);
        }
}