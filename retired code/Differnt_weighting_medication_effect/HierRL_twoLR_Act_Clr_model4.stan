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
    int<lower=1> nMeds;                         // Number of Medication level
    int<lower=1> nConds;                        // Number of condition, Action and Color value learning
    array[N] int<lower=1, upper=2> medication;  // 1 indecates OFF medication and 2 indicates On medication
    array[N] int<lower=1, upper=2> condition;   // 1 indecates first codition (Action) and 2 indicates second condition (Color)
    real<lower=0, upper=1> p_push_init;         // Initial value of reward probability for pushed responce
    real<lower=0, upper=1> p_yell_init;         // Initial value of reward probability for Color responce
}
parameters {
    /* Hierarchical mu parameter*/                               
    array[nMeds, nConds] real hier_alphaAct_mu;    // Mean Hierarchical Learning rate for action Learning Value
    array[nMeds, nConds] real hier_alphaClr_mu;    // Mean Hierarchical Learning rate for color Learning Value
    array[nMeds] real hier_weight_mu_Act;                       // Mean Hierarchical Wieghtening of Action Learning Value  and medication effect
    real hier_weight_mu_Clr;          // Mean Hierarchical Wieghtening of Color Learning Value
    array[nMeds, nConds] real hier_sensitivity_mu; // Mean Hierarchical snesitivity, With a higher sensitivity value θ, choices are more sensitive to value differences

    /* Hierarchical sd parameter*/                               
    real<lower=0> hier_alpha_sd;      // Between-participant variability Learning rate for Learning Value
    real<lower=0> hier_weight_sd;     // Between-participant variability Wieghtening of Action Learning Value against to Color Learnig Value
    real<lower=0> hier_sensitivity_sd;   // Between-participant variability sensitivity

    /* participant-level main paameter*/
    array[nParts, nMeds, nConds] real z_alphaAct;    // Learning rate for Action Learning Value
    array[nParts, nMeds, nConds] real z_alphaClr;    // Learning rate for Color Learning Value
    array[nParts, nMeds] real z_weight_Act;         // Wieghtening of Action Learning Value  and medication effect
    array[nParts] real z_weight_Clr;          // Wieghtening of Color Learning Value
    array[nParts, nMeds, nConds] real z_sensitivity; // With a higher sensitivity value θ, choices are more sensitive to value differences

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
    array[nParts, nMeds, nConds] real<lower=0, upper=1> transfer_alphaAct;   // Learning rate for Action Learning Value
    array[nParts, nMeds, nConds] real<lower=0, upper=1> transfer_alphaClr;   // Learning rate for Color Learning Value
    array[nParts, nMeds] real<lower=0, upper=1> transfer_weight_Act;  // Wieghtening of Action Learning Value  and medication effect
    array[nParts] real<lower=0, upper=1> transfer_weight_Clr;  // Wieghtening of Action Learning Value
    array[nParts, nMeds, nConds] real<lower=0> transfer_sensitivity;         // With a higher sensitivity value θ, choices are more sensitive to value differences
    
    /* Transfer Hierarchical parameters just for output*/
    array[nMeds, nConds] real<lower=0, upper=1> transfer_hier_alphaAct_mu;   // Hierarchical Learning rate for Action Learning Value
    array[nMeds, nConds] real<lower=0, upper=1> transfer_hier_alphaClr_mu;   // Hierarchical Learning rate for Color Learning Value
    array[nMeds] real<lower=0, upper=1> transfer_hier_weight_mu_Act;  // Hierarchical Wieghtening of Action Learning Value  
    real<lower=0, upper=1> transfer_hier_weight_mu_Clr;  // Hierarchical Wieghtening of Color vluea learning and medication effect
    array[nMeds, nConds] real<lower=0> transfer_hier_sensitivity_mu;         // Hierarchical snesitivity, With a higher sensitivity value θ, choices are more sensitive to value differences

	transfer_hier_alphaAct_mu = Phi(hier_alphaAct_mu);				// for the output
	transfer_hier_alphaClr_mu = Phi(hier_alphaClr_mu);				 
    transfer_hier_weight_mu_Act = Phi(hier_weight_mu_Act);
    transfer_hier_weight_mu_Clr = Phi(hier_weight_mu_Clr);
	for (g in 1:nMeds){
        for (c in 1:nConds){
            transfer_hier_sensitivity_mu[g, c] = log(1 + exp(hier_sensitivity_mu[g, c]));
        }
    }

    for (p in 1:nParts) {
        transfer_weight_Clr[p] = Phi(hier_weight_mu_Clr + z_weight_Clr[p]*hier_weight_sd);
        for (g in 1:nMeds){
            transfer_weight_Act[p, g] = Phi(hier_weight_mu_Act[g] + z_weight_Act[p, g]*hier_weight_sd);
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
       
        // Relative contribution of Color Value Learning
        if (condition[i]==1){
            EV_push_yell = transfer_weight_Act[participant[i], medication[i]]*EV_push + (1 - transfer_weight_Act[participant[i], medication[i]])*EV_yell;
            EV_push_blue = transfer_weight_Act[participant[i], medication[i]]*EV_push + (1 - transfer_weight_Act[participant[i], medication[i]])*EV_blue;
            EV_pull_yell = transfer_weight_Act[participant[i], medication[i]]*EV_pull + (1 - transfer_weight_Act[participant[i], medication[i]])*EV_yell;
            EV_pull_blue = transfer_weight_Act[participant[i], medication[i]]*EV_pull + (1 - transfer_weight_Act[participant[i], medication[i]])*EV_blue;
        }
        // Relative contribution of Color Value Learning
        if (condition[i]==2){
            EV_push_yell = transfer_weight_Clr[participant[i]]*EV_push + (1 - transfer_weight_Clr[participant[i]])*EV_yell;
            EV_push_blue = transfer_weight_Clr[participant[i]]*EV_push + (1 - transfer_weight_Clr[participant[i]])*EV_blue;
            EV_pull_yell = transfer_weight_Clr[participant[i]]*EV_pull + (1 - transfer_weight_Clr[participant[i]])*EV_yell;
            EV_pull_blue = transfer_weight_Clr[participant[i]]*EV_pull + (1 - transfer_weight_Clr[participant[i]])*EV_blue;
        }
       
        /* Calculating the soft-max function over weightening Action and Color medications*/ 
        // pushed and yellow vs pulled and blue
        if ((pushed[i] == 1 && yellowChosen[i] == 1) || (pushed[i] == 0 && yellowChosen[i] == 0))
            soft_max_EV[i] = exp(transfer_sensitivity[participant[i], medication[i], condition[i]]*EV_push_yell)/(exp(transfer_sensitivity[participant[i], medication[i], condition[i]]*EV_push_yell) + exp(transfer_sensitivity[participant[i], medication[i], condition[i]]*EV_pull_blue));

        // pushed and blue vs pulled and yellow
        if ((pushed[i] == 1 && yellowChosen[i] == 0) || (pushed[i] == 0 && yellowChosen[i] == 1))
            soft_max_EV[i] = exp(transfer_sensitivity[participant[i], medication[i], condition[i]]*EV_push_blue)/(exp(transfer_sensitivity[participant[i], medication[i], condition[i]]*EV_push_blue) + exp(transfer_sensitivity[participant[i], medication[i], condition[i]]*EV_pull_yell));  
          
        // RL rule update for computing prediction error and internal value expectation for the next trial based on the current reward output and interal value expectation
        /*Action value learning*/
        if (pushed[i] == 1){
            p_push = p_push + transfer_alphaAct[participant[i], medication[i], condition[i]]*(rewarded[i] - p_push); 
        }
        else{
            p_pull = p_pull + transfer_alphaAct[participant[i], medication[i], condition[i]]*(rewarded[i] - p_pull);
        }   

        /*Color value learning*/
        if (yellowChosen[i] == 1){
           p_yell = p_yell + transfer_alphaClr[participant[i], medication[i], condition[i]]*(rewarded[i] - p_yell);
        }    
        else{
           p_blue = p_blue + transfer_alphaClr[participant[i], medication[i], condition[i]]*(rewarded[i] - p_blue);
        }
    }   
}
model { 
    /* Hierarchical mu parameter*/    
    hier_weight_mu_Clr ~ normal(0,1);
    for (g in 1:nMeds){
        hier_weight_mu_Act[g] ~ normal(0,1);
        for (c in 1:nConds){
            hier_alphaAct_mu[g,c] ~ normal(0,1);
            hier_alphaClr_mu[g,c] ~ normal(0,1);
            hier_sensitivity_mu[g,c] ~ normal(1,5); 
        }
    }

    /* Hierarchical sd parameter*/
    hier_alpha_sd ~ normal(0,.1) T[0,];  
    hier_weight_sd ~ normal(0,.1) T[0,]; 
    hier_sensitivity_sd ~ normal(0,.1) T[0,];
    
    /* participant-level main paameter*/
    for (p in 1:nParts) {
        z_weight_Clr[p] ~ normal(0,1);
        for (g in 1:nMeds){
            z_weight_Act[p, g] ~ normal(0,1);
            for (c in 1:nConds){
                z_alphaAct[p, g, c] ~ normal(0,1);
                z_alphaClr[p, g, c] ~ normal(0,1);
                z_sensitivity[p, g, c] ~ normal(0,1); 
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