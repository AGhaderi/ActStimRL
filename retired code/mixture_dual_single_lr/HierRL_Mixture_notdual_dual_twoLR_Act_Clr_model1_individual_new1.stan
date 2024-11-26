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
    array[N] int<lower=1, upper=2> condition;   // 1 indecates first codition (Action) and 2 indicates second condition (Color)
    real<lower=0, upper=1> p_push_init;         // Initial value of reward probability for pushed responce
    real<lower=0, upper=1> p_yell_init;         // Initial value of reward probability for Color responce
}
parameters {
    /* Hierarchical mu parameter model1*/                               
    array[nConds] real hier_alphaAct_pos_mu1;   // Mean Hierarchical Positive Learning rate for action Learning Value and medication_session effect
    array[nConds] real hier_alphaAct_neg_mu1;   // Mean Hierarchical Negative Learning rate for action Learning Value and medication_session effect
    array[nConds] real hier_alphaClr_pos_mu1;   // Mean Hierarchical Positive Learning rate for color Learning Value and medication_session effect
    array[nConds] real hier_alphaClr_neg_mu1;   // Mean Hierarchical Negative Learning rate for color Learning Value and medication_session effect
    array[nConds] real hier_weight_mu1;         // Mean Hierarchical Wieghtening of Action Learning Value against to Color Learnig Value
    array[nConds] real hier_sensitivity_mu1;    // Mean Hierarchical snesitivity, With a higher sensitivity value θ, choices are more sensitive to value differences

    /* Hierarchical sd parameter model1*/                               
    real<lower=0> hier_alpha_sd1;      // Between-participant variability Learning rate for Learning Value
    real<lower=0> hier_weight_sd1;     // Between-participant variability Wieghtening of Action Learning Value against to Color Learnig Value
    real<lower=0> hier_sensitivity_sd1;   // Between-participant variability sensitivity

    /* participant-level main paameter model1*/
    array[nParts, nConds] real z_alphaAct_pos1;   // Positive Learning rate for Action Learning Value
    array[nParts, nConds] real z_alphaAct_neg1;   // Negative Learning rate for Action Learning Value
    array[nParts, nConds] real z_alphaClr_pos1;   // Positive Learning rate for Color Learning Value
    array[nParts, nConds] real z_alphaClr_neg1;   // Negative Learning rate for Color Learning Value
    array[nParts, nConds] real z_weight1;         // Wieghtening of Action Learning Value against to Learnig Value
    array[nParts, nConds] real z_sensitivity1;    // With a higher sensitivity value θ, choices are more sensitive to value differences
    
    /* Hierarchical mu parameter model2*/                               
    array[nConds] real hier_alphaAct_mu2;    // Mean Hierarchical Learning rate for action Learning Value and medication_session effect
    array[nConds] real hier_alphaClr_mu2;    // Mean Hierarchical Learning rate for color Learning Value and medication_session effect
    array[nConds] real hier_weight_mu2;   // Mean Hierarchical Wieghtening of Action Learning Value against to Color Learnig Value
    array[nConds] real hier_sensitivity_mu2;    // Mean Hierarchical snesitivity, With a higher sensitivity value θ, choices are more sensitive to value differences

    /* Hierarchical sd parameter model2*/                               
    real<lower=0> hier_alpha_sd2;      // Between-participant variability Learning rate for Learning Value
    real<lower=0> hier_weight_sd2;     // Between-participant variability Wieghtening of Action Learning Value against to Color Learnig Value
    real<lower=0> hier_sensitivity_sd2;   // Between-participant variability sensitivity

    /* participant-level main paameter model2*/
    array[nParts, nConds] real z_alphaAct2;   // Learning rate for Action Learning Value
    array[nParts, nConds] real z_alphaClr2;   // Learning rate for Color Learning Value
    array[nParts, nConds] real z_weight2;  // Wieghtening of Action Learning Value against to Learnig Value
    array[nParts, nConds] real z_sensitivity2;         // With a higher sensitivity value θ, choices are more sensitive to value differences
    
    real<lower=0, upper=1> theta;          // mixing proportions
}
transformed parameters {
    /* probability of each features and their combination */
    real p_push1;   // Probability of reward for pushing responce
    real p_pull1;   // Probability of reward for pulling responce
    real p_yell1;   // Probability of reward for yrllow responce
    real p_blue1;   // Probability of reward for blue responce
    real EV_push1;  // Standard Expected Value of push action
    real EV_pull1;  // Standard Expected Value of pull action
    real EV_yell1;  // Standard Expected Value of yellow action
    real EV_blue1;  // Standard Expected Value of blue action
    real EV_push_yell1;      // Weighting two strategies between push action and yellow color values learning
    real EV_push_blue1;      // Weighting two strategies between push action and blue color values learning
    real EV_pull_yell1;      // Weighting two strategies between pull action and yellow color values learning
    real EV_pull_blue1;      // Weighting two strategies between pull action and blue color values learning
    array[N] real soft_max_EV1;  //  The soft-max function for each trial, trial-by-trial probability for model 1
    
    /* probability of each features and their combination */
    real p_push2;   // Probability of reward for pushing responce
    real p_pull2;   // Probability of reward for pulling responce
    real p_yell2;   // Probability of reward for yrllow responce
    real p_blue2;   // Probability of reward for blue responce
    real EV_push2;  // Standard Expected Value of push action
    real EV_pull2;  // Standard Expected Value of pull action
    real EV_yell2;  // Standard Expected Value of yellow action
    real EV_blue2;  // Standard Expected Value of blue action
    real EV_push_yell2;      // Weighting two strategies between push action and yellow color values learning
    real EV_push_blue2;      // Weighting two strategies between push action and blue color values learning
    real EV_pull_yell2;      // Weighting two strategies between pull action and yellow color values learning
    real EV_pull_blue2;      // Weighting two strategies between pull action and blue color values learning
    array[N] real soft_max_EV2;  //  The soft-max function for each trial, trial-by-trial probability for model 2
   
    /* Transfer individual parameters for model1 */
    array[nParts, nConds] real<lower=0, upper=1> transfer_alphaAct_pos1;   // Poistive Learning rate for Action Learning Value
    array[nParts, nConds] real<lower=0, upper=1> transfer_alphaAct_neg1;   // Negative Learning rate for Action Learning Value
    array[nParts, nConds] real<lower=0, upper=1> transfer_alphaClr_pos1;   // Positive Learning rate for Color Learning Value
    array[nParts, nConds] real<lower=0, upper=1> transfer_alphaClr_neg1;   // Negative Learning rate for Color Learning Value
    array[nParts, nConds] real<lower=0, upper=1> transfer_weight1;  // Wieghtening of Action Learning Value against to Color Learnig Value
    array[nParts, nConds] real<lower=0> transfer_sensitivity1;         // With a higher sensitivity value θ, choices are more sensitive to value differences
    
    /* Transfer Hierarchical parameters just for output for model1*/
    array[nConds] real<lower=0, upper=1> transfer_hier_alphaAct_pos_mu1;   // Hierarchical Positive Learning rate for Action Learning Value
    array[nConds] real<lower=0, upper=1> transfer_hier_alphaAct_neg_mu1;   // Hierarchical Negative Learning rate for Action Learning Value
    array[nConds] real<lower=0, upper=1> transfer_hier_alphaClr_pos_mu1;   // Hierarchical Positive Learning rate for Color Learning Value
    array[nConds] real<lower=0, upper=1> transfer_hier_alphaClr_neg_mu1;   // Hierarchical Negative  Learning rate for Color Learning Value
    array[nConds] real<lower=0, upper=1> transfer_hier_weight_mu1;  // Hierarchical Wieghtening of Action Learning Value against to Color Learnig Value
    array[nConds] real<lower=0> transfer_hier_sensitivity_mu1;         // Hierarchical snesitivity, With a higher sensitivity value θ, choices are more sensitive to value differences

	transfer_hier_alphaAct_pos_mu1 = Phi(hier_alphaAct_pos_mu1);				// for the output
	transfer_hier_alphaAct_neg_mu1 = Phi(hier_alphaAct_neg_mu1);				 
	transfer_hier_alphaClr_pos_mu1 = Phi(hier_alphaClr_pos_mu1);				 
	transfer_hier_alphaClr_neg_mu1 = Phi(hier_alphaClr_neg_mu1);	
    transfer_hier_weight_mu1 = Phi(hier_weight_mu1);
    for (c in 1:nConds){
        transfer_hier_sensitivity_mu1[c] = log(1 + exp(hier_sensitivity_mu1[c]));
    }

    for (p in 1:nParts) {
        for (c in 1:nConds){
            transfer_weight1[p, c] = Phi(hier_weight_mu1[c] + z_weight1[p, c]*hier_weight_sd1);
            transfer_alphaAct_pos1[p, c] = Phi(hier_alphaAct_pos_mu1[c] + z_alphaAct_pos1[p, c]*hier_alpha_sd1);
            transfer_alphaAct_neg1[p, c] = Phi(hier_alphaAct_neg_mu1[c] + z_alphaAct_neg1[p, c]*hier_alpha_sd1);
            transfer_alphaClr_pos1[p, c] = Phi(hier_alphaClr_pos_mu1[c] + z_alphaClr_pos1[p, c]*hier_alpha_sd1);
            transfer_alphaClr_neg1[p, c] = Phi(hier_alphaClr_neg_mu1[c] + z_alphaClr_neg1[p, c]*hier_alpha_sd1);
            transfer_sensitivity1[p, c] = log(1 + exp(hier_sensitivity_mu1[c] + z_sensitivity1[p, c]*hier_sensitivity_sd1));
        }
    }


  /* Transfer individual parameters for model 2*/
    array[nParts, nConds] real<lower=0, upper=1> transfer_alphaAct2;   // Learning rate for Action Learning Value
    array[nParts, nConds] real<lower=0, upper=1> transfer_alphaClr2;   // Learning rate for Color Learning Value
    array[nParts, nConds] real<lower=0, upper=1> transfer_weight2;  // Wieghtening of Action Learning Value against to Color Learnig Value
    array[nParts, nConds] real<lower=0> transfer_sensitivity2;         // With a higher sensitivity value θ, choices are more sensitive to value differences
    
    /* Transfer Hierarchical parameters just for output*/
    array[nConds] real<lower=0, upper=1> transfer_hier_alphaAct_mu2;   // Hierarchical Learning rate for Action Learning Value
    array[nConds] real<lower=0, upper=1> transfer_hier_alphaClr_mu2;   // Hierarchical Learning rate for Color Learning Value
    array[nConds] real<lower=0, upper=1> transfer_hier_weight_mu2;  // Hierarchical Wieghtening of Action Learning Value against to Color Learnig Value
    array[nConds] real<lower=0> transfer_hier_sensitivity_mu2;         // Hierarchical snesitivity, With a higher sensitivity value θ, choices are more sensitive to value differences

	transfer_hier_alphaAct_mu2 = Phi(hier_alphaAct_mu2);				// for the output
	transfer_hier_alphaClr_mu2 = Phi(hier_alphaClr_mu2);				 
    transfer_hier_weight_mu2 = Phi(hier_weight_mu2);
    for (c in 1:nConds){
        transfer_hier_sensitivity_mu2[c] = log(1 + exp(hier_sensitivity_mu2[c]));
    }

    for (p in 1:nParts) {
        for (c in 1:nConds){
            transfer_weight2[p, c] = Phi(hier_weight_mu2[c] + z_weight2[p, c]*hier_weight_sd2);
            transfer_alphaAct2[p, c] = Phi(hier_alphaAct_mu2[c] + z_alphaAct2[p, c]*hier_alpha_sd2);
            transfer_alphaClr2[p, c] = Phi(hier_alphaClr_mu2[c] + z_alphaClr2[p, c]*hier_alpha_sd2);
            transfer_sensitivity2[p, c] = log(1 + exp(hier_sensitivity_mu2[c] + z_sensitivity2[p,c]*hier_sensitivity_sd2));
        }
    }
 
    // Calculating the probability of reward for model 1
    for (i in 1:N) {
        // Restart probability of variable for each environemnt and condition
        if (indicator[i]==1){
            p_push1 = p_push_init;
            p_pull1 = 1 - p_push_init;
            p_yell1 = p_yell_init;
            p_blue1 = 1 - p_yell_init;
        }
        // Calculating the Standard Expected Value
        EV_push1 = p_push1*winAmtPushable[i];
        EV_pull1 = p_pull1*winAmtPullable[i];
        EV_yell1 = p_yell1*winAmtYellow[i];
        EV_blue1 = p_blue1*winAmtBlue[i];

        // Relative contribution of Action Value Learning verus Color Value Learning
        EV_push_yell1 = transfer_weight1[participant[i], condition[i]]*EV_push1 + (1 - transfer_weight1[participant[i], condition[i]])*EV_yell1;
        EV_push_blue1 = transfer_weight1[participant[i], condition[i]]*EV_push1 + (1 - transfer_weight1[participant[i], condition[i]])*EV_blue1;
        EV_pull_yell1 = transfer_weight1[participant[i], condition[i]]*EV_pull1 + (1 - transfer_weight1[participant[i], condition[i]])*EV_yell1;
        EV_pull_blue1 = transfer_weight1[participant[i], condition[i]]*EV_pull1 + (1 - transfer_weight1[participant[i], condition[i]])*EV_blue1;
       
        /* Calculating the soft-max function over weightening Action and Color conditions*/ 
        // pushed/yellow coded and pulled/blue coded 1
        if ((pushed[i] == 1 && yellowChosen[i] == 1) || (pushed[i] == 0 && yellowChosen[i] == 0))
            soft_max_EV1[i] = exp(transfer_sensitivity1[participant[i], condition[i]]*EV_push_yell1)/(exp(transfer_sensitivity1[participant[i], condition[i]]*EV_push_yell1) + exp(transfer_sensitivity1[participant[i], condition[i]]*EV_pull_blue1));

        //  pushed/blue coded 1 and pulled/yellow coded 0
        if ((pushed[i] == 1 && yellowChosen[i] == 0) || (pushed[i] == 0 && yellowChosen[i] == 1))
            soft_max_EV1[i] = exp(transfer_sensitivity1[participant[i], condition[i]]*EV_push_blue1)/(exp(transfer_sensitivity1[participant[i], condition[i]]*EV_push_blue1) + exp(transfer_sensitivity1[participant[i], condition[i]]*EV_pull_yell1));  
          
        // change the probability of 1 to near to 1, to avoid further exception in Model block  
        if (soft_max_EV1[i] == 1){
            soft_max_EV1[i] = 0.99999999999999;
        }

        // RL rule update for computing prediction error and internal value expectation for the next trial based on the current reward output and interal value expectation
        /*Action value learning*/
        if (pushed[i] == 1){
            // positive RPE
            if((rewarded[i] - p_push1 )>=0){ 
                p_push1 = p_push1 + transfer_alphaAct_pos1[participant[i], condition[i]]*(rewarded[i] - p_push1);
            } 
            // negative RPE
            else{
                p_push1 = p_push1 + transfer_alphaAct_neg1[participant[i], condition[i]]*(rewarded[i] - p_push1); 
            }
        }
        else{
            // positive RPE
            if((rewarded[i] - p_pull1 )>=0){ 
                p_pull1 = p_pull1 + transfer_alphaAct_pos1[participant[i], condition[i]]*(rewarded[i] - p_pull1);
            } 
            // negative RPE
            else{
                p_pull1 = p_pull1 + transfer_alphaAct_neg1[participant[i], condition[i]]*(rewarded[i] - p_pull1);
            }
        }   

        /*Color value learning*/
        if (yellowChosen[i] == 1){
            // positive RPE
            if((rewarded[i] - p_yell1)>=0 ){ 
                p_yell1 = p_yell1 + transfer_alphaClr_pos1[participant[i], condition[i]]*(rewarded[i] - p_yell1);
            } 
            // negative RPE
            else{
                p_yell1 = p_yell1 + transfer_alphaClr_neg1[participant[i], condition[i]]*(rewarded[i] - p_yell1);
            }
        }    
        else{
            // positive RPE
            if((rewarded[i] - p_blue1)>=0){ 
                p_blue1 = p_blue1 + transfer_alphaClr_pos1[participant[i], condition[i]]*(rewarded[i] - p_blue1);
            } 
            // negative RPE
            else{
                p_blue1 = p_blue1 + transfer_alphaClr_neg1[participant[i], condition[i]]*(rewarded[i] - p_blue1);
            }
        }
    }   

    // Calculating the probability of reward
    for (i in 1:N) {
        // Restart probability of variable for each environemnt and condition
        if (indicator[i]==1){
            p_push2 = p_push_init;
            p_pull2 = 1 - p_push_init;
            p_yell2 = p_yell_init;
            p_blue2 = 1 - p_yell_init;
        }
        // Calculating the Standard Expected Value
        EV_push2 = p_push2*winAmtPushable[i];
        EV_pull2 = p_pull2*winAmtPullable[i];
        EV_yell2 = p_yell2*winAmtYellow[i];
        EV_blue2 = p_blue2*winAmtBlue[i];
       
        // Relative contribution of Action Value Learning verus Color Value Learning
        EV_push_yell2 = transfer_weight2[participant[i], condition[i]]*EV_push2 + (1 - transfer_weight2[participant[i], condition[i]])*EV_yell2;
        EV_push_blue2 = transfer_weight2[participant[i], condition[i]]*EV_push2 + (1 - transfer_weight2[participant[i], condition[i]])*EV_blue2;
        EV_pull_yell2 = transfer_weight2[participant[i], condition[i]]*EV_pull2 + (1 - transfer_weight2[participant[i], condition[i]])*EV_yell2;
        EV_pull_blue2 = transfer_weight2[participant[i], condition[i]]*EV_pull2 + (1 - transfer_weight2[participant[i], condition[i]])*EV_blue2;
       
        /* Calculating the soft-max function over weightening Action and Color conditions*/ 
        // pushed and yellow vs pulled and blue
        if ((pushed[i] == 1 && yellowChosen[i] == 1) || (pushed[i] == 0 && yellowChosen[i] == 0))
            soft_max_EV2[i] = exp(transfer_sensitivity2[participant[i], condition[i]]*EV_push_yell2)/(exp(transfer_sensitivity2[participant[i], condition[i]]*EV_push_yell2) + exp(transfer_sensitivity2[participant[i], condition[i]]*EV_pull_blue2));

        // pushed and blue vs pulled and yellow
        if ((pushed[i] == 1 && yellowChosen[i] == 0) || (pushed[i] == 0 && yellowChosen[i] == 1))
            soft_max_EV2[i] = exp(transfer_sensitivity2[participant[i], condition[i]]*EV_push_blue2)/(exp(transfer_sensitivity2[participant[i], condition[i]]*EV_push_blue2) + exp(transfer_sensitivity2[participant[i], condition[i]]*EV_pull_yell2));  

       // change the probability of 1 to near to 1, to avoid further exception in Model block  
        if (soft_max_EV2[i]==1){
            soft_max_EV2[i] =  0.99999999999999;
        }
        
        // RL rule update for computing prediction error and internal value expectation for the next trial based on the current reward output and interal value expectation
        /*Action value learning*/
        if (pushed[i] == 1){
            p_push2 = p_push2 + transfer_alphaAct2[participant[i], condition[i]]*(rewarded[i] - p_push2); 
        }
        else{
            p_pull2 = p_pull2 + transfer_alphaAct2[participant[i], condition[i]]*(rewarded[i] - p_pull2);
        }   

        /*Color value learning*/
        if (yellowChosen[i] == 1){
           p_yell2 = p_yell2 + transfer_alphaClr2[participant[i], condition[i]]*(rewarded[i] - p_yell2);
        }    
        else{
           p_blue2 = p_blue2 + transfer_alphaClr2[participant[i], condition[i]]*(rewarded[i] - p_blue2);
        }
    }   

}
model { 
    /* Hierarchical mu parameter for model 1*/    
    for (c in 1:nConds){
        hier_weight_mu1[c] ~ normal(0,1);
        hier_alphaAct_pos_mu1[c] ~ normal(0,1);
        hier_alphaAct_neg_mu1[c] ~ normal(0,1);
        hier_alphaClr_pos_mu1[c] ~ normal(0,1);
        hier_alphaClr_neg_mu1[c] ~ normal(0,1);
        hier_sensitivity_mu1[c] ~ normal(1,5);
    }

    /* Hierarchical sd parameter foe model 1*/
    hier_alpha_sd1 ~ normal(0,1);  
    hier_weight_sd1 ~ normal(0,1); 
    hier_sensitivity_sd1 ~ normal(0,1);

    /* participant-level main paameter for model 1*/
    for (p in 1:nParts) {
        for (c in 1:nConds){
            z_weight1[p, c] ~ normal(0,1);
            z_alphaAct_pos1[p, c] ~ normal(0,1);
            z_alphaAct_neg1[p, c] ~ normal(0,1);
            z_alphaClr_pos1[p, c] ~ normal(0,1);
            z_alphaClr_neg1[p, c] ~ normal(0,1);
            z_sensitivity1[p, c] ~ normal(0,1); 
        }
    }

    /* Hierarchical mu parameter for model 2*/    
    for (c in 1:nConds){
        hier_weight_mu2[c] ~ normal(0,1);
        hier_alphaAct_mu2[c] ~ normal(0,1);
        hier_alphaClr_mu2[c] ~ normal(0,1);
        hier_sensitivity_mu2[c] ~ normal(1,5); 
    }

    /* Hierarchical sd parameter*/
    hier_alpha_sd2 ~ normal(0,1);  
    hier_weight_sd2 ~ normal(0,1); 
    hier_sensitivity_sd2 ~ normal(0,1);
    
    /* participant-level main paameter*/
    for (p in 1:nParts) {
        for (c in 1:nConds){
            z_weight2[p, c] ~ normal(0,1);
            z_alphaAct2[p, c] ~ normal(0,1);
            z_alphaClr2[p, c] ~ normal(0,1);
            z_sensitivity2[p, c] ~ normal(0,1); 
        }
    }

    /* participant-level main paameter for model 1*/
    theta ~ beta(1,1); 

    /* RL likelihood */
    for (i in 1:N) { 
        vector[2] lps;
        lps[1] = log(theta) + bernoulli_lpmf(pushed[i] | soft_max_EV1[i]);  // first term of mixture 
        lps[2] = log1m(theta) + bernoulli_lpmf(pushed[i] | soft_max_EV2[i]);  // second term of mixture
        target += log_sum_exp(lps);
    }
}