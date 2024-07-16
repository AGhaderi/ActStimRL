/* Model 7 focus on just probability between action and color in action value learning condition
*/ 

data {
    int<lower=1> N;        // Number of trial-level observations
    int<lower=1> nParts;   // Number of participants
    array[N] int<lower=0, upper=1> pushed;             // 1 if pushed and 0 if pulled 
    array[N] int<lower=0, upper=1> yellowChosen;       // 1 if yellow color is chosen and 0 if yellow color is not chosen 
    array[N] int<lower=0, upper=1> rewarded;           // 1 for rewarding and 0 for punishment
    array[N] int<lower=1> participant;                 // participant index for each trial
    array[N] int<lower=1> indicator;                   // indicator of the first trial of each participant, the first is denoted 1 otherwise 0
    real<lower=0, upper=1> p_push_init;         // Initial value of reward probability for pushed responce
    real<lower=0, upper=1> p_yell_init;         // Initial value of reward probability for Color responce
}
parameters {
    /* Hierarchical mu parameter*/                               
    real hier_alphaAct_mu;    // Mean Hierarchical Learning rate for Action Learning Value
    real hier_alphaClr_mu;    // Mean Hierarchical Learning rate for Color Learning Value
    real hier_weightAct_mu;   // Mean Hierarchical Wieghtening of Action Learning Value against to Color Learnig Value
    real hier_sensitivity_mu;    // Mean Hierarchical snesitivity, With a higher sensitivity value θ, choices are more sensitive to value differences

    /* Hierarchical sd parameter*/                               
    real<lower=0> hier_alphaAct_sd;      // Between-participant variability Learning rate for Action Learning Value
    real<lower=0> hier_alphaClr_sd;      // Between-participant variability Learning rate for Color Learning Value
    real<lower=0> hier_weightAct_sd;     // Between-participant variability Wieghtening of Action Learning Value against to Color Learnig Value
    real<lower=0> hier_sensitivity_sd;   // Between-participant variability sensitivity

    /* participant-level main paameter*/
    array[nParts] real z_alphaAct;   // Learning rate for Action Learning Value
    array[nParts] real z_alphaClr;   // Learning rate for Color Learning Value
    array[nParts] real z_weightAct;  // Wieghtening of Action Learning Value against to Color Learnig Value
    array[nParts] real z_sensitivity;         // With a higher sensitivity value θ, choices are more sensitive to value differences

}
transformed parameters {
    /* probability of each features and their combination */
    real p_push;   // Probability of reward for pushing responce
    real p_pull;   // Probability of reward for pulling responce
    real p_yell;   // Probability of reward for yrllow responce
    real p_blue;   // Probability of reward for blue responce
    vector[N] soft_max_act;  //  The soft-max function for each trial for Acion values
    vector[N] soft_max_clr;  //  The soft-max function for each trial for color values
   
    /* Transfer individual parameters */
    array[nParts] real<lower=0, upper=1> transfer_alphaAct;   // Learning rate for Action Learning Value
    array[nParts] real<lower=0, upper=1> transfer_alphaClr;   // Learning rate for Color Learning Value
    array[nParts] real<lower=0, upper=1> transfer_weightAct;  // Wieghtening of Action Learning Value against to Color Learnig Value
    array[nParts] real<lower=0> transfer_sensitivity;         // With a higher sensitivity value θ, choices are more sensitive to value differences
    
    /* Transfer Hierarchical parameters just for output*/
    real<lower=0, upper=1> transfer_hier_alphaAct_mu;   // Hierarchical Learning rate for Action Learning Value
    real<lower=0, upper=1> transfer_hier_alphaClr_mu;   // Hierarchical Learning rate for Color Learning Value
    real<lower=0, upper=1> transfer_hier_weightAct_mu;  // Hierarchical Wieghtening of Action Learning Value against to Color Learnig Value
    real<lower=0> transfer_hier_sensitivity_mu;         // Hierarchical snesitivity, With a higher sensitivity value θ, choices are more sensitive to value differences

	transfer_hier_alphaAct_mu = Phi(hier_alphaAct_mu);				// for the output
    transfer_hier_alphaClr_mu = Phi(hier_alphaClr_mu);
    transfer_hier_weightAct_mu = Phi(hier_weightAct_mu);
    transfer_hier_sensitivity_mu = log(1 + exp(hier_sensitivity_mu));

    for (p in 1:nParts) {
        transfer_alphaAct[p] = Phi(hier_alphaAct_mu + z_alphaAct[p]*hier_alphaAct_sd);
        transfer_alphaClr[p] = Phi(hier_alphaClr_mu + z_alphaClr[p]*hier_alphaClr_sd);
        transfer_weightAct[p] = Phi(hier_weightAct_mu + z_weightAct[p]*hier_weightAct_sd);
        transfer_sensitivity[p] = log(1 + exp(hier_sensitivity_mu + z_sensitivity[p]*hier_sensitivity_sd));
    }

    // Calculating the probability of reward
   for (i in 1:N) {
        if (indicator[i]==1){
            p_push = p_push_init;
            p_pull = 1 - p_push_init;
            p_yell = p_yell_init;
            p_blue = 1 - p_yell_init;
        }
        
        // Action-based decision  
        // soft max function
        soft_max_act[i] = exp(transfer_sensitivity[participant[i]]*p_push)/(exp(transfer_sensitivity[participant[i]]*p_push) + exp(transfer_sensitivity[participant[i]]*p_pull));

        // Color-based decision  
        // soft max function 
        soft_max_clr[i] = exp(transfer_sensitivity[participant[i]]*p_yell)/(exp(transfer_sensitivity[participant[i]]*p_yell) + exp(transfer_sensitivity[participant[i]]*p_blue));  
   
        // RL rule update for computing prediction error and internal value expectation for the next trial based on the current reward output and interal value expectation
        if (pushed[i] == 1){
            p_push = p_push + transfer_alphaAct[participant[i]]*(rewarded[i] - p_push); 
            p_pull = 1 - p_push;
        }
        else{
            p_pull = p_pull + transfer_alphaAct[participant[i]]*(rewarded[i] - p_pull);
            p_push = 1 - p_pull;
        }    
        if (yellowChosen[i] == 1){
           p_yell = p_yell + transfer_alphaClr[participant[i]]*(rewarded[i] - p_yell);
           p_blue = 1 - p_yell;
        }    
        else{
           p_blue = p_blue + transfer_alphaClr[participant[i]]*(rewarded[i] - p_blue);
           p_yell = 1 - p_blue;           
        }
       
    }   
}
model { 
    /* Hierarchical mu parameter*/    
    hier_alphaAct_mu ~ normal(0,1);
    hier_alphaClr_mu ~ normal(0,1); 
    hier_sensitivity_mu ~ normal(0,5); 
    hier_weightAct_mu ~ normal(0,1);
 
    /* Hierarchical sd parameter*/
    hier_alphaAct_sd ~ normal(1,2);  
    hier_alphaClr_sd ~ normal(1,2);
    hier_weightAct_sd ~ normal(1,2); 
    hier_sensitivity_sd ~ normal(1,2);
    
    /* participant-level main paameter*/
    for (p in 1:nParts) {
        z_alphaAct[p] ~ normal(0,1); 
        z_alphaClr[p] ~ normal(0,1);
        z_weightAct[p] ~ normal(0,1);
        z_sensitivity[p] ~ normal(0,1); 
    }

    /* RL likelihood */
    for (i in 1:N) { 
        // Relative contribution of Action Value Learning verus Color Value Learning
        target += log_sum_exp(log(transfer_weightAct[participant[i]]) + binomial_lpmf(pushed[i] | 1, soft_max_act[i]),
                              log1m(transfer_weightAct[participant[i]]) + binomial_lpmf(yellowChosen[i] | 1, soft_max_clr[i]));
    }
}
generated quantities { 
   vector[N] log_lik;  
    /*  RL Log density likelihood */
    for (i in 1:N) {
        log_lik[i] = log_sum_exp(log(transfer_weightAct[participant[i]]) + binomial_lpmf(pushed[i] | 1, soft_max_act[i]),
                              log1m(transfer_weightAct[participant[i]]) + binomial_lpmf(yellowChosen[i] | 1, soft_max_clr[i]));
        }
}