/*
Model 3 in Table 1.
RL(w, +alpha, -alpha, beta)
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
    int<lower=1> nConds;                        // Number of condition, Action and Color value learning
    array[N] int<lower=1, upper=2> condition;   // 1 indecates first condition (Action) and 2 indicates second condition (Color)
    int<lower=1> nMeds_nSes;                           // Number of medication_session level (OFF vs ON)
    array[N] int<lower=1, upper=2> medication_session; // 1 indecates OFF medication_session and 2 indicates On medication_session
}
parameters {
    /* Hierarchical mu parameter*/                               
    real hier_alpha_pos_mu;    // Mean Hierarchical Positive Learning rate
    array[nMeds_nSes ] real hier_alpha_neg_mu;    // Mean Hierarchical Negative Learning rate 
    array[nConds, nMeds_nSes ] real hier_weight_mu;       // Mean Hierarchical Weighting 
    array[nMeds_nSes ] real hier_sensitivity_mu;  // Mean Hierarchical snesitivity
    
    /* Hierarchical sd parameter*/                               
    real<lower=0> hier_alpha_sd;      // Between-participant variability Learning rate
    real<lower=0> hier_weight_sd;     // Between-participant variability Wieghtening
    real<lower=0> hier_sensitivity_sd;   // Between-participant variability sensitivity

    /* participant-level main paameter*/
    array[nParts] real z_alpha_pos;   // Positive Learning rate
    array[nParts, nMeds_nSes ] real z_alpha_neg;   // Negative Learning rate
    array[nParts, nConds, nMeds_nSes ] real z_weight;  // Wieghtening
    array[nParts, nMeds_nSes ] real z_sensitivity;         // Sensitivity  

}
transformed parameters {
    /* probability of each features and their combination */
    real p_push;   // Probability of reward for pushing responce
    real p_yell;   // Probability of reward for yrllow responce
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
    array[nParts] real<lower=0, upper=1> transfer_alpha_pos;   // Poistive Learning rate  
    array[nParts, nMeds_nSes ] real<lower=0, upper=1> transfer_alpha_neg;   // Negative Learning rate  
    array[nParts, nConds, nMeds_nSes ] real<lower=0, upper=1> transfer_weight;  // Wieghtening  
    array[nParts, nMeds_nSes ] real<lower=0> transfer_sensitivity;         // Sensitivity 
    
    /* Transfer Hierarchical parameters just for output*/
    real<lower=0, upper=1> transfer_hier_alpha_pos_mu;   // Hierarchical Positive Learning rate
    array[nMeds_nSes ] real<lower=0, upper=1> transfer_hier_alpha_neg_mu;   // Hierarchical Negative Learning rate
    array[nConds, nMeds_nSes ] real<lower=0, upper=1> transfer_hier_weight_mu;  // Hierarchical Wieghtening
    array[nMeds_nSes ] real<lower=0> transfer_hier_sensitivity_mu;         // Hierarchical snesitivity

	transfer_hier_alpha_pos_mu = inv_logit(hier_alpha_pos_mu);				// for the output
	transfer_hier_alpha_neg_mu = inv_logit(hier_alpha_neg_mu);				 
    transfer_hier_weight_mu = inv_logit(hier_weight_mu);
    for (s in 1:nMeds_nSes){
        transfer_hier_sensitivity_mu[s] = log(1 + exp(hier_sensitivity_mu[s]));
    }

    for (p in 1:nParts) {
        for (c in 1:nConds){
            for (s in 1:nMeds_nSes){
                transfer_weight[p,c,s] = inv_logit(hier_weight_mu[c,s] + z_weight[p,c,s]*hier_weight_sd);
            }   
        }
    }

    for (p in 1:nParts) {
        transfer_alpha_pos[p] = inv_logit(hier_alpha_pos_mu + z_alpha_pos[p]*hier_alpha_sd);
        for (s in 1:nMeds_nSes){
            transfer_alpha_neg[p,s] = inv_logit(hier_alpha_neg_mu[s] + z_alpha_neg[p,s]*hier_alpha_sd);
            transfer_sensitivity[p,s] = log(1 + exp(hier_sensitivity_mu[s] + z_sensitivity[p,s]*hier_sensitivity_sd));
        }
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
       
        // Relative contribution of ion Value Learning verus Color Value Learning
        EV_push_yell = transfer_weight[participant[i], condition[i], medication_session[i]]*EV_push + (1 - transfer_weight[participant[i], condition[i], medication_session[i]])*EV_yell;
        EV_push_blue = transfer_weight[participant[i], condition[i], medication_session[i]]*EV_push + (1 - transfer_weight[participant[i], condition[i], medication_session[i]])*EV_blue;
        EV_pull_yell = transfer_weight[participant[i], condition[i], medication_session[i]]*EV_pull + (1 - transfer_weight[participant[i], condition[i], medication_session[i]])*EV_yell;
        EV_pull_blue = transfer_weight[participant[i], condition[i], medication_session[i]]*EV_pull + (1 - transfer_weight[participant[i], condition[i], medication_session[i]])*EV_blue;
       
        /* Calculating the soft-max function over weightening Action and Color conditions*/ 
        // pushed/yellow coded and pulled/blue coded 1
        if ((pushed[i] == 1 && yellowChosen[i] == 1) || (pushed[i] == 0 && yellowChosen[i] == 0))
            soft_max_EV[i] = exp(transfer_sensitivity[participant[i], medication_session[i]]*EV_push_yell)/(exp(transfer_sensitivity[participant[i], medication_session[i]]*EV_push_yell) + exp(transfer_sensitivity[participant[i], medication_session[i]]*EV_pull_blue));

        //  pushed/blue coded 1 and pulled/yellow coded 0
        else if ((pushed[i] == 1 && yellowChosen[i] == 0) || (pushed[i] == 0 && yellowChosen[i] == 1))
            soft_max_EV[i] = exp(transfer_sensitivity[participant[i], medication_session[i]]*EV_push_blue)/(exp(transfer_sensitivity[participant[i], medication_session[i]]*EV_push_blue) + exp(transfer_sensitivity[participant[i], medication_session[i]]*EV_pull_yell));  
          
        // RL rule update for computing prediction error and internal value expectation for the next trial based on the current reward output and interal value expectation
        /*Action value learning*/
        if (pushed[i] == 1){
            // positive RPE
            if((rewarded[i] - p_push)>=0 ){ 
                p_push = p_push + transfer_alpha_pos[participant[i]]*(rewarded[i] - p_push);
            } 
            // negative RPE
            else{
                p_push = p_push + transfer_alpha_neg[participant[i], medication_session[i]]*(rewarded[i] - p_push); 
            }
        }
        else{
            // positive RPE
            if((rewarded[i] + p_push - 1)>=0){ 
                p_push = p_push - transfer_alpha_pos[participant[i]]*(rewarded[i] + p_push - 1);
            } 
            // negative RPE
            else{
                p_push = p_push - transfer_alpha_neg[participant[i], medication_session[i]]*(rewarded[i] + p_push - 1);
            }
        }   

        /*Color value learning*/
        if (yellowChosen[i] == 1){
            // positive RPE
            if((rewarded[i] - p_yell)>=0){ 
                p_yell = p_yell + transfer_alpha_pos[participant[i]]*(rewarded[i] - p_yell);
            } 
            // negative RPE
            else{
                p_yell = p_yell + transfer_alpha_neg[participant[i], medication_session[i]]*(rewarded[i] - p_yell);
            }
        }    
        else{
            // positive RPE
            if((rewarded[i] + p_yell - 1)>=0){ 
                p_yell = p_yell - transfer_alpha_pos[participant[i]]*(rewarded[i] + p_yell - 1);
            } 
            // negative RPE
            else{
                p_yell = p_yell - transfer_alpha_neg[participant[i], medication_session[i]]*(rewarded[i] + p_yell - 1);
            }
        }
    }   
}
model { 
    /* Hierarchical mu parameter*/
    for (c in 1:nConds){
        for (s in 1:nMeds_nSes){
            hier_weight_mu[c,s] ~ normal(0,2);
        }
    }
    hier_alpha_pos_mu ~ normal(0,2);
    for (s in 1:nMeds_nSes){
        hier_alpha_neg_mu[s] ~ normal(0,2);
        hier_sensitivity_mu[s] ~ normal(0,3); 
    }

    /* Hierarchical sd parameter*/
    hier_alpha_sd ~ normal(0,.5);  
    hier_weight_sd ~ normal(0,.5); 
    hier_sensitivity_sd ~ normal(0,.5);
    
    /* participant-level main paameter*/
    for (p in 1:nParts) {
        for (c in 1:nConds){
            for (s in 1:nMeds_nSes){
                z_weight[p,c,s] ~ normal(0,1);
            }
        }
    }
    
    for (p in 1:nParts) {
        z_alpha_pos[p] ~ normal(0,1);
        for (s in 1:nMeds_nSes){
            z_alpha_neg[p,s] ~ normal(0,1);
            z_sensitivity[p,s] ~ normal(0,1); 
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