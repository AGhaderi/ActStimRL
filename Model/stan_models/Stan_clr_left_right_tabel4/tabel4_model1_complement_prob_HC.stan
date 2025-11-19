/*
Model 3 in Table 1.
RL(w, +alpha, -alpha, beta)
*/

data {
    int<lower=1> N;        // Number of trial-level observations
    int<lower=1> nParts;   // Number of participants
    array[N] int<lower=0, upper=1> leftChosen;           // 1 if left and 0 if right 
    array[N] int<lower=0, upper=1> yellowChosen;       // 1 if yellow color is chosen and 0 if yellow color is not chosen 
    array[N] real<lower=0, upper=100> winAmtLeft;  // The amount of values feedback when left is correct response
    array[N] real<lower=0, upper=100> winAmtRight;  // The amount of values feedback when right is correct response
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
    array[nMeds_nSes] real hier_alpha_pos_mu;    // Mean Hierarchical Positive Learning rate
    array[nMeds_nSes] real hier_alpha_neg_mu;    // Mean Hierarchical Negative Learning rate 
    array[nConds, nMeds_nSes] real hier_weight_mu;       // Mean Hierarchical Weighting 
    array[nConds, nMeds_nSes] real hier_sensitivity_mu;  // Mean Hierarchical snesitivity
    
    /* Hierarchical sd parameter*/                               
    real<lower=0> hier_alpha_sd;      // Between-participant variability Learning rate
    real<lower=0> hier_weight_sd;     // Between-participant variability Wieghtening
    real<lower=0> hier_sensitivity_sd;   // Between-participant variability sensitivity

    /* participant-level main paameter*/
    array[nParts, nMeds_nSes] real z_alpha_pos;   // Positive Learning rate
    array[nParts, nMeds_nSes] real z_alpha_neg;   // Negative Learning rate
    array[nParts, nConds, nMeds_nSes] real z_weight;  // Wieghtening
    array[nParts, nConds, nMeds_nSes] real z_sensitivity;         // Sensitivity  

}
transformed parameters {
    /* probability of each features and their combination */
    real p_left;   // Probability of reward for left responce
    real p_yell;   // Probability of reward for yrllow responce
    real EV_left;  // Standard Expected Value of left action
    real EV_right;  // Standard Expected Value of right action
    real EV_yell;  // Standard Expected Value of yellow action
    real EV_blue;  // Standard Expected Value of blue action
    real EV_left_yell;      // Weighting two strategies between left action and yellow color values learning
    real EV_left_blue;      // Weighting two strategies between left action and blue color values learning
    real EV_right_yell;      // Weighting two strategies between right action and yellow color values learning
    real EV_right_blue;      // Weighting two strategies between right action and blue color values learning
    vector[N] soft_max_EV;  //  The soft-max function for each trial, trial-by-trial probability
   
    /* Transfer individual parameters */
    array[nParts, nMeds_nSes] real<lower=0, upper=1> transfer_alpha_pos;   // Poistive Learning rate  
    array[nParts, nMeds_nSes] real<lower=0, upper=1> transfer_alpha_neg;   // Negative Learning rate  
    array[nParts, nConds, nMeds_nSes] real<lower=0, upper=1> transfer_weight;  // Wieghtening  
    array[nParts, nConds, nMeds_nSes] real<lower=0> transfer_sensitivity;         // Sensitivity 
    
    /* Transfer Hierarchical parameters just for output*/
    array[nMeds_nSes] real<lower=0, upper=1> transfer_hier_alpha_pos_mu;   // Hierarchical Positive Learning rate
    array[nMeds_nSes] real<lower=0, upper=1> transfer_hier_alpha_neg_mu;   // Hierarchical Negative Learning rate
    array[nConds, nMeds_nSes] real<lower=0, upper=1> transfer_hier_weight_mu;  // Hierarchical Wieghtening
    array[nConds, nMeds_nSes] real<lower=0> transfer_hier_sensitivity_mu;         // Hierarchical snesitivity

	transfer_hier_alpha_pos_mu = inv_logit(hier_alpha_pos_mu);				// for the output
	transfer_hier_alpha_neg_mu = inv_logit(hier_alpha_neg_mu);				 
    transfer_hier_weight_mu = inv_logit(hier_weight_mu);
    for (c in 1:nConds){
        for (s in 1:nMeds_nSes){
            transfer_hier_sensitivity_mu[c,s] = log(1 + exp(hier_sensitivity_mu[c,s]));
        }
    }

    for (p in 1:nParts) {
        for (c in 1:nConds){
            for (s in 1:nMeds_nSes){
                transfer_weight[p,c,s] = inv_logit(hier_weight_mu[c,s] + z_weight[p,c,s]*hier_weight_sd);
                transfer_sensitivity[p,c,s] = log(1 + exp(hier_sensitivity_mu[c,s] + z_sensitivity[p,c,s]*hier_sensitivity_sd));
            }   
        }
    }
    for (p in 1:nParts) {
        for (s in 1:nMeds_nSes){
            transfer_alpha_pos[p,s] = inv_logit(hier_alpha_pos_mu[s] + z_alpha_pos[p,s]*hier_alpha_sd);
            transfer_alpha_neg[p,s] = inv_logit(hier_alpha_neg_mu[s] + z_alpha_neg[p,s]*hier_alpha_sd);
        }
    }

    // Calculating the probability of reward
   for (i in 1:N) {
        // Restart probability of variable for each environemnt and condition
        if (indicator[i]==1){
            p_left = .5;
            p_yell = .5;
        }
        // Calculating the Standard Expected Value
        EV_left = p_left*winAmtLeft[i];
        EV_right = (1-p_left)*winAmtRight[i];
        EV_yell = p_yell*winAmtYellow[i];
        EV_blue = (1-p_yell)*winAmtBlue[i];
       
        // Relative contribution of ion Value Learning verus Color Value Learning
        EV_left_yell = transfer_weight[participant[i], condition[i], medication_session[i]]*EV_left + (1 - transfer_weight[participant[i], condition[i], medication_session[i]])*EV_yell;
        EV_left_blue = transfer_weight[participant[i], condition[i], medication_session[i]]*EV_left + (1 - transfer_weight[participant[i], condition[i], medication_session[i]])*EV_blue;
        EV_right_yell = transfer_weight[participant[i], condition[i], medication_session[i]]*EV_right + (1 - transfer_weight[participant[i], condition[i], medication_session[i]])*EV_yell;
        EV_right_blue = transfer_weight[participant[i], condition[i], medication_session[i]]*EV_right + (1 - transfer_weight[participant[i], condition[i], medication_session[i]])*EV_blue;
       
        /* Calculating the soft-max function over weightening Action and Color conditions*/ 
        // left/yellow coded and right/blue coded 1
        if ((leftChosen[i] == 1 && yellowChosen[i] == 1) || (leftChosen[i] == 0 && yellowChosen[i] == 0))
            soft_max_EV[i] = exp(transfer_sensitivity[participant[i], condition[i], medication_session[i]]*EV_left_yell)/(exp(transfer_sensitivity[participant[i], condition[i], medication_session[i]]*EV_left_yell) + exp(transfer_sensitivity[participant[i], condition[i], medication_session[i]]*EV_right_blue));

        //  left/blue coded 1 and right/yellow coded 0
        else if ((leftChosen[i] == 1 && yellowChosen[i] == 0) || (leftChosen[i] == 0 && yellowChosen[i] == 1))
            soft_max_EV[i] = exp(transfer_sensitivity[participant[i], condition[i], medication_session[i]]*EV_left_blue)/(exp(transfer_sensitivity[participant[i], condition[i], medication_session[i]]*EV_left_blue) + exp(transfer_sensitivity[participant[i], condition[i], medication_session[i]]*EV_right_yell));  
          
        // RL rule update for computing prediction error and internal value expectation for the next trial based on the current reward output and interal value expectation
        /*Action value learning*/
        if (leftChosen[i] == 1){
            // positive RPE
            if((rewarded[i] - p_left)>=0 ){ 
                p_left = p_left + transfer_alpha_pos[participant[i], medication_session[i]]*(rewarded[i] - p_left);
            } 
            // negative RPE
            else{
                p_left = p_left + transfer_alpha_neg[participant[i], medication_session[i]]*(rewarded[i] - p_left); 
            }
        }
        else{
            // positive RPE
            if((rewarded[i] + p_left - 1)>=0){ 
                p_left = p_left - transfer_alpha_pos[participant[i], medication_session[i]]*(rewarded[i] + p_left - 1);
            } 
            // negative RPE
            else{
                p_left = p_left - transfer_alpha_neg[participant[i], medication_session[i]]*(rewarded[i] + p_left - 1);
            }
        }   

        /*Color value learning*/
        if (yellowChosen[i] == 1){
            // positive RPE
            if((rewarded[i] - p_yell)>=0){ 
                p_yell = p_yell + transfer_alpha_pos[participant[i], medication_session[i]]*(rewarded[i] - p_yell);
            } 
            // negative RPE
            else{
                p_yell = p_yell + transfer_alpha_neg[participant[i], medication_session[i]]*(rewarded[i] - p_yell);
            }
        }    
        else{
            // positive RPE
            if((rewarded[i] + p_yell - 1)>=0){ 
                p_yell = p_yell - transfer_alpha_pos[participant[i], medication_session[i]]*(rewarded[i] + p_yell - 1);
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
            hier_sensitivity_mu[c,s] ~ normal(0,3); 
        }
    }
    for (s in 1:nMeds_nSes){
        hier_alpha_pos_mu[s] ~ normal(0,2);
        hier_alpha_neg_mu[s] ~ normal(0,2);
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
                z_sensitivity[p,c,s] ~ normal(0,1); 
            }
        }
    }
    for (p in 1:nParts) {
        for (s in 1:nMeds_nSes){
            z_alpha_pos[p,s] ~ normal(0,1);
            z_alpha_neg[p,s] ~ normal(0,1);
        }
    }

    /* RL likelihood */
    for (i in 1:N) { 
        leftChosen[i] ~ bernoulli(soft_max_EV[i]);
        }
}
generated quantities { 
   vector[N] log_lik;  
    /*  RL Log density likelihood */
    for (i in 1:N) {
        log_lik[i] = bernoulli_lpmf(leftChosen[i] | soft_max_EV[i]);
    }
}