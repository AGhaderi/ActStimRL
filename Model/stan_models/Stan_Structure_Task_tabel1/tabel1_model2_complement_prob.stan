/*
Model 2 in Table 1.
RL(alpha, beta)
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
}
parameters {
    /* Hierarchical mu parameter*/                               
    real hier_alpha_mu;    // Mean Hierarchical Learning rate for both action and color Learning Value
    real hier_sensitivity_mu;    // Mean Hierarchical snesitivity

    /* Hierarchical sd parameter*/                               
    real<lower=0> hier_alpha_sd;      // Between-participant variability Learning rate for Learning Value
    real<lower=0> hier_sensitivity_sd;   // Between-participant variability sensitivity

    /* participant-level main paameter*/
    array[nParts] real z_alpha;      // Individual Learning rate for action and Color Learning Value
    array[nParts] real z_sensitivity; // Individual sensitivity parameter

}
transformed parameters {
    /* probability of each features and their combination */
    real p_push;   // Probability of reward for pushing responce
    real p_yell;   // Probability of reward for yrllow responce
    real EV_push;  // Standard Expected Value of push action
    real EV_pull;  // Standard Expected Value of pull action
    real EV_yell;  // Standard Expected Value of yellow action
    real EV_blue;  // Standard Expected Value of blue action
    vector[N] soft_max_Act;  //  The soft-max function for each trial for Action value learing
    vector[N] soft_max_Clr;  //  The soft-max function for each trial for Color value learning
   
    /* Transfer individual parameters */
    array[nParts] real<lower=0, upper=1> transfer_alpha;    // Learning rate for Action and Color Learning Value
    array[nParts] real<lower=0> transfer_sensitivity;       // sensitivity paramter
    
    /* Transfer Hierarchical parameters just for output*/
    real<lower=0, upper=1> transfer_hier_alpha_mu;   // Hierarchical Learning rate for Color and Action Learning Value
    real<lower=0> transfer_hier_sensitivity_mu;         // Hierarchical snesitivity

	transfer_hier_alpha_mu = inv_logit(hier_alpha_mu);				// for the output
    transfer_hier_sensitivity_mu = log(1 + exp(hier_sensitivity_mu));

    for (p in 1:nParts) {
        transfer_alpha[p] = inv_logit(hier_alpha_mu + z_alpha[p]*hier_alpha_sd);
        transfer_sensitivity[p] = log(1 + exp(hier_sensitivity_mu + z_sensitivity[p]*hier_sensitivity_sd));
    }

    // Calculating the probability of reward
   for (i in 1:N) {
        // Restart probability of variable for each environemnt and condition
        if (indicator[i]==1){
            p_push = .5;
            p_yell = .5;
        }
        // Calculating the Standard Expected Value
        EV_push = p_push*winAmtPushable[i];
        EV_pull = (1-p_push)*winAmtPullable[i];
        EV_yell = p_yell*winAmtYellow[i];
        EV_blue = (1-p_yell)*winAmtBlue[i];
       
        /* Calculating the soft-max function*/ 
        // pushed  vs pulled
        soft_max_Act[i] = exp(transfer_sensitivity[participant[i]]*EV_push)/(exp(transfer_sensitivity[participant[i]]*EV_push) + exp(transfer_sensitivity[participant[i]]*EV_pull));
        
        // yellow vs blue 
        soft_max_Clr[i] = exp(transfer_sensitivity[participant[i]]*EV_yell)/(exp(transfer_sensitivity[participant[i]]*EV_yell) + exp(transfer_sensitivity[participant[i]]*EV_blue));  
          
        // RL rule update for computing prediction error and internal value expectation for the next trial based on the current reward output and interal value expectation
        /*Action value learning*/
        if (pushed[i] == 1){
            p_push = p_push + transfer_alpha[participant[i]]*(rewarded[i] - p_push); 
        }
        else{
            p_push = p_push - transfer_alpha[participant[i]]*(rewarded[i] + p_push - 1);
        }   

        /*Color value learning*/
        if (yellowChosen[i] == 1){
           p_yell = p_yell + transfer_alpha[participant[i]]*(rewarded[i] - p_yell);
        }    
        else{
           p_yell = p_yell - transfer_alpha[participant[i]]*(rewarded[i] + p_yell - 1);
        }
    }   
}
model { 
    /* Hierarchical mu parameter*/    
    hier_alpha_mu ~ normal(0,2);
    hier_sensitivity_mu ~ normal(0,3); 

    /* Hierarchical sd parameter*/
    hier_alpha_sd ~ normal(0,.5);  
    hier_sensitivity_sd ~ normal(0,.5);
    
    /* participant-level main paameter*/
    for (p in 1:nParts) {
        z_alpha[p] ~ normal(0,1);
        z_sensitivity[p] ~ normal(0,1); 
    }

    /* RL likelihood */
    for (i in 1:N) { 
        pushed[i] ~ bernoulli(soft_max_Act[i]);
        yellowChosen[i] ~ bernoulli(soft_max_Clr[i]);
        }
}
generated quantities { 
   vector[N] log_lik;  
    /*  RL Log density likelihood */
    for (i in 1:N) {
        log_lik[i] = bernoulli_lpmf(pushed[i] | soft_max_Act[i]) + bernoulli_lpmf(yellowChosen[i] | soft_max_Clr[i]);
        }
}