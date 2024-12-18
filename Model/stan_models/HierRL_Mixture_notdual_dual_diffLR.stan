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
    /* Hierarchical mu parameter model1*/                               
    array[nMeds_nSes, nConds] real hier_alphaAct_pos_mu1;   // Mean Hierarchical Positive Learning rate for action Learning Value and medication_session effect
    array[nMeds_nSes, nConds] real hier_alphaAct_neg_mu1;   // Mean Hierarchical Negative Learning rate for action Learning Value and medication_session effect
    array[nMeds_nSes, nConds] real hier_alphaClr_pos_mu1;   // Mean Hierarchical Positive Learning rate for color Learning Value and medication_session effect
    array[nMeds_nSes, nConds] real hier_alphaClr_neg_mu1;   // Mean Hierarchical Negative Learning rate for color Learning Value and medication_session effect
    array[nMeds_nSes, nConds] real hier_weight_mu;         // Mean Hierarchical Wieghtening of Action Learning Value against to Color Learnig Value
    array[nMeds_nSes, nConds] real hier_sensitivity_mu;    // Mean Hierarchical snesitivity, With a higher sensitivity value θ, choices are more sensitive to value differences

    /* Hierarchical sd parameter model1*/                               
    real<lower=0> hier_alpha_sd1;      // Between-participant variability Learning rate for Learning Value
    real<lower=0> hier_weight_sd;     // Between-participant variability Wieghtening of Action Learning Value against to Color Learnig Value
    real<lower=0> hier_sensitivity_sd;   // Between-participant variability sensitivity

    /* participant-level main paameter model1*/
    array[nParts, nMeds_nSes, nConds] real z_alphaAct_pos1;   // Positive Learning rate for Action Learning Value
    array[nParts, nMeds_nSes, nConds] real z_alphaAct_neg1;   // Negative Learning rate for Action Learning Value
    array[nParts, nMeds_nSes, nConds] real z_alphaClr_pos1;   // Positive Learning rate for Color Learning Value
    array[nParts, nMeds_nSes, nConds] real z_alphaClr_neg1;   // Negative Learning rate for Color Learning Value
    array[nParts, nMeds_nSes, nConds] real z_weight;         // Wieghtening of Action Learning Value against to Learnig Value
    array[nParts, nMeds_nSes, nConds] real z_sensitivity;    // With a higher sensitivity value θ, choices are more sensitive to value differences
    
    /* Hierarchical mu parameter model2*/                               
    array[nMeds_nSes, nConds] real hier_alphaAct_mu2;    // Mean Hierarchical Learning rate for action Learning Value and medication_session effect
    array[nMeds_nSes, nConds] real hier_alphaClr_mu2;    // Mean Hierarchical Learning rate for color Learning Value and medication_session effect

    /* Hierarchical sd parameter model2*/                               
    real<lower=0> hier_alpha_sd2;      // Between-participant variability Learning rate for Learning Value

    /* participant-level main paameter model2*/
    array[nParts, nMeds_nSes, nConds] real z_alphaAct2;   // Learning rate for Action Learning Value
    array[nParts, nMeds_nSes, nConds] real z_alphaClr2;   // Learning rate for Color Learning Value
    
    /* mixture model between model1 and model2*/
    real hier_theta_mu;          // Mean Hierarchical snesitivity mixing proportions
    array[nParts] real z_theta;          // mixing proportions
    real<lower=0> hier_theta_sd;        // Between-participant variability mixing proportions

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
    array[nParts, nMeds_nSes, nConds] real<lower=0, upper=1> transfer_alphaAct_pos1;   // Poistive Learning rate for Action Learning Value
    array[nParts, nMeds_nSes, nConds] real<lower=0, upper=1> transfer_alphaAct_neg1;   // Negative Learning rate for Action Learning Value
    array[nParts, nMeds_nSes, nConds] real<lower=0, upper=1> transfer_alphaClr_pos1;   // Positive Learning rate for Color Learning Value
    array[nParts, nMeds_nSes, nConds] real<lower=0, upper=1> transfer_alphaClr_neg1;   // Negative Learning rate for Color Learning Value
    array[nParts, nMeds_nSes, nConds] real<lower=0, upper=1> transfer_weight;  // Wieghtening of Action Learning Value against to Color Learnig Value
    array[nParts, nMeds_nSes, nConds] real<lower=0> transfer_sensitivity;         // With a higher sensitivity value θ, choices are more sensitive to value differences
    
    /* Transfer Hierarchical parameters just for output for model1*/
    array[nMeds_nSes, nConds] real<lower=0, upper=1> transfer_hier_alphaAct_pos_mu1;   // Hierarchical Positive Learning rate for Action Learning Value
    array[nMeds_nSes, nConds] real<lower=0, upper=1> transfer_hier_alphaAct_neg_mu1;   // Hierarchical Negative Learning rate for Action Learning Value
    array[nMeds_nSes, nConds] real<lower=0, upper=1> transfer_hier_alphaClr_pos_mu1;   // Hierarchical Positive Learning rate for Color Learning Value
    array[nMeds_nSes, nConds] real<lower=0, upper=1> transfer_hier_alphaClr_neg_mu1;   // Hierarchical Negative  Learning rate for Color Learning Value
    array[nMeds_nSes, nConds] real<lower=0, upper=1> transfer_hier_weight_mu;  // Hierarchical Wieghtening of Action Learning Value against to Color Learnig Value
    array[nMeds_nSes, nConds] real<lower=0> transfer_hier_sensitivity_mu;         // Hierarchical snesitivity, With a higher sensitivity value θ, choices are more sensitive to value differences

	transfer_hier_alphaAct_pos_mu1 = inv_logit(hier_alphaAct_pos_mu1);				// for the output
	transfer_hier_alphaAct_neg_mu1 = inv_logit(hier_alphaAct_neg_mu1);				 
	transfer_hier_alphaClr_pos_mu1 = inv_logit(hier_alphaClr_pos_mu1);				 
	transfer_hier_alphaClr_neg_mu1 = inv_logit(hier_alphaClr_neg_mu1);	
    transfer_hier_weight_mu = inv_logit(hier_weight_mu);
	for (g in 1:nMeds_nSes){
        for (c in 1:nConds){
            transfer_hier_sensitivity_mu[g, c] = log(1 + exp(hier_sensitivity_mu[g, c]));
        }
    }

    for (p in 1:nParts) {
        for (g in 1:nMeds_nSes){
            for (c in 1:nConds){
                transfer_weight[p, g, c] = inv_logit(hier_weight_mu[g, c] + z_weight[p, g, c]*hier_weight_sd);
                transfer_alphaAct_pos1[p, g, c] = inv_logit(hier_alphaAct_pos_mu1[g, c] + z_alphaAct_pos1[p, g, c]*hier_alpha_sd1);
                transfer_alphaAct_neg1[p, g, c] = inv_logit(hier_alphaAct_neg_mu1[g, c] + z_alphaAct_neg1[p, g, c]*hier_alpha_sd1);
                transfer_alphaClr_pos1[p, g, c] = inv_logit(hier_alphaClr_pos_mu1[g, c] + z_alphaClr_pos1[p, g, c]*hier_alpha_sd1);
                transfer_alphaClr_neg1[p, g, c] = inv_logit(hier_alphaClr_neg_mu1[g, c] + z_alphaClr_neg1[p, g, c]*hier_alpha_sd1);
                transfer_sensitivity[p, g, c] = log(1 + exp(hier_sensitivity_mu[g, c] + z_sensitivity[p,g, c]*hier_sensitivity_sd));
            }
        }
    }


  /* Transfer individual parameters for model 2*/
    array[nParts, nMeds_nSes, nConds] real<lower=0, upper=1> transfer_alphaAct2;   // Learning rate for Action Learning Value
    array[nParts, nMeds_nSes, nConds] real<lower=0, upper=1> transfer_alphaClr2;   // Learning rate for Color Learning Value
    
    /* Transfer Hierarchical parameters just for output*/
    array[nMeds_nSes, nConds] real<lower=0, upper=1> transfer_hier_alphaAct_mu2;   // Hierarchical Learning rate for Action Learning Value
    array[nMeds_nSes, nConds] real<lower=0, upper=1> transfer_hier_alphaClr_mu2;   // Hierarchical Learning rate for Color Learning Value

	transfer_hier_alphaAct_mu2 = inv_logit(hier_alphaAct_mu2);				// for the output
	transfer_hier_alphaClr_mu2 = inv_logit(hier_alphaClr_mu2);				 

    for (p in 1:nParts) {
        for (g in 1:nMeds_nSes){
            for (c in 1:nConds){
                transfer_alphaAct2[p, g, c] = inv_logit(hier_alphaAct_mu2[g, c] + z_alphaAct2[p, g, c]*hier_alpha_sd2);
                transfer_alphaClr2[p, g, c] = inv_logit(hier_alphaClr_mu2[g, c] + z_alphaClr2[p, g, c]*hier_alpha_sd2);
            }
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
        EV_push_yell1 = transfer_weight[participant[i], medication_session[i], condition[i]]*EV_push1 + (1 - transfer_weight[participant[i], medication_session[i], condition[i]])*EV_yell1;
        EV_push_blue1 = transfer_weight[participant[i], medication_session[i], condition[i]]*EV_push1 + (1 - transfer_weight[participant[i], medication_session[i], condition[i]])*EV_blue1;
        EV_pull_yell1 = transfer_weight[participant[i], medication_session[i], condition[i]]*EV_pull1 + (1 - transfer_weight[participant[i], medication_session[i], condition[i]])*EV_yell1;
        EV_pull_blue1 = transfer_weight[participant[i], medication_session[i], condition[i]]*EV_pull1 + (1 - transfer_weight[participant[i], medication_session[i], condition[i]])*EV_blue1;
       
        /* Calculating the soft-max function over weightening Action and Color conditions*/ 
        // pushed/yellow coded and pulled/blue coded 1
        if ((pushed[i] == 1 && yellowChosen[i] == 1) || (pushed[i] == 0 && yellowChosen[i] == 0))
            soft_max_EV1[i] = exp(transfer_sensitivity[participant[i], medication_session[i], condition[i]]*EV_push_yell1)/(exp(transfer_sensitivity[participant[i], medication_session[i], condition[i]]*EV_push_yell1) + exp(transfer_sensitivity[participant[i], medication_session[i], condition[i]]*EV_pull_blue1));

        //  pushed/blue coded 1 and pulled/yellow coded 0
        if ((pushed[i] == 1 && yellowChosen[i] == 0) || (pushed[i] == 0 && yellowChosen[i] == 1))
            soft_max_EV1[i] = exp(transfer_sensitivity[participant[i], medication_session[i], condition[i]]*EV_push_blue1)/(exp(transfer_sensitivity[participant[i], medication_session[i], condition[i]]*EV_push_blue1) + exp(transfer_sensitivity[participant[i], medication_session[i], condition[i]]*EV_pull_yell1));  
          
        // change the probability of 1 to near to 1, to avoid further exception in Model block  
        if (soft_max_EV1[i] == 1){
            soft_max_EV1[i] = 0.99999999999999;
        }

        // RL rule update for computing prediction error and internal value expectation for the next trial based on the current reward output and interal value expectation
        /*Action value learning*/
        if (pushed[i] == 1){
            // positive RPE
            if((rewarded[i] - p_push1 )>=0){ 
                p_push1 = p_push1 + transfer_alphaAct_pos1[participant[i], medication_session[i], condition[i]]*(rewarded[i] - p_push1);
            } 
            // negative RPE
            else{
                p_push1 = p_push1 + transfer_alphaAct_neg1[participant[i], medication_session[i], condition[i]]*(rewarded[i] - p_push1); 
            }
        }
        else{
            // positive RPE
            if((rewarded[i] - p_pull1 )>=0){ 
                p_pull1 = p_pull1 + transfer_alphaAct_pos1[participant[i], medication_session[i], condition[i]]*(rewarded[i] - p_pull1);
            } 
            // negative RPE
            else{
                p_pull1 = p_pull1 + transfer_alphaAct_neg1[participant[i], medication_session[i], condition[i]]*(rewarded[i] - p_pull1);
            }
        }   

        /*Color value learning*/
        if (yellowChosen[i] == 1){
            // positive RPE
            if((rewarded[i] - p_yell1)>=0 ){ 
                p_yell1 = p_yell1 + transfer_alphaClr_pos1[participant[i], medication_session[i], condition[i]]*(rewarded[i] - p_yell1);
            } 
            // negative RPE
            else{
                p_yell1 = p_yell1 + transfer_alphaClr_neg1[participant[i], medication_session[i], condition[i]]*(rewarded[i] - p_yell1);
            }
        }    
        else{
            // positive RPE
            if((rewarded[i] - p_blue1)>=0){ 
                p_blue1 = p_blue1 + transfer_alphaClr_pos1[participant[i], medication_session[i], condition[i]]*(rewarded[i] - p_blue1);
            } 
            // negative RPE
            else{
                p_blue1 = p_blue1 + transfer_alphaClr_neg1[participant[i], medication_session[i], condition[i]]*(rewarded[i] - p_blue1);
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
        EV_push_yell2 = transfer_weight[participant[i], medication_session[i], condition[i]]*EV_push2 + (1 - transfer_weight[participant[i], medication_session[i], condition[i]])*EV_yell2;
        EV_push_blue2 = transfer_weight[participant[i], medication_session[i], condition[i]]*EV_push2 + (1 - transfer_weight[participant[i], medication_session[i], condition[i]])*EV_blue2;
        EV_pull_yell2 = transfer_weight[participant[i], medication_session[i], condition[i]]*EV_pull2 + (1 - transfer_weight[participant[i], medication_session[i], condition[i]])*EV_yell2;
        EV_pull_blue2 = transfer_weight[participant[i], medication_session[i], condition[i]]*EV_pull2 + (1 - transfer_weight[participant[i], medication_session[i], condition[i]])*EV_blue2;
       
        /* Calculating the soft-max function over weightening Action and Color conditions*/ 
        // pushed and yellow vs pulled and blue
        if ((pushed[i] == 1 && yellowChosen[i] == 1) || (pushed[i] == 0 && yellowChosen[i] == 0))
            soft_max_EV2[i] = exp(transfer_sensitivity[participant[i], medication_session[i], condition[i]]*EV_push_yell2)/(exp(transfer_sensitivity[participant[i], medication_session[i], condition[i]]*EV_push_yell2) + exp(transfer_sensitivity[participant[i], medication_session[i], condition[i]]*EV_pull_blue2));

        // pushed and blue vs pulled and yellow
        if ((pushed[i] == 1 && yellowChosen[i] == 0) || (pushed[i] == 0 && yellowChosen[i] == 1))
            soft_max_EV2[i] = exp(transfer_sensitivity[participant[i], medication_session[i], condition[i]]*EV_push_blue2)/(exp(transfer_sensitivity[participant[i], medication_session[i], condition[i]]*EV_push_blue2) + exp(transfer_sensitivity[participant[i], medication_session[i], condition[i]]*EV_pull_yell2));  

       // change the probability of 1 to near to 1, to avoid further exception in Model block  
        if (soft_max_EV2[i]==1){
            soft_max_EV2[i] =  0.99999999999999;
        }
        
        // RL rule update for computing prediction error and internal value expectation for the next trial based on the current reward output and interal value expectation
        /*Action value learning*/
        if (pushed[i] == 1){
            p_push2 = p_push2 + transfer_alphaAct2[participant[i], medication_session[i], condition[i]]*(rewarded[i] - p_push2); 
        }
        else{
            p_pull2 = p_pull2 + transfer_alphaAct2[participant[i], medication_session[i], condition[i]]*(rewarded[i] - p_pull2);
        }   

        /*Color value learning*/
        if (yellowChosen[i] == 1){
           p_yell2 = p_yell2 + transfer_alphaClr2[participant[i], medication_session[i], condition[i]]*(rewarded[i] - p_yell2);
        }    
        else{
           p_blue2 = p_blue2 + transfer_alphaClr2[participant[i], medication_session[i], condition[i]]*(rewarded[i] - p_blue2);
        }
    }   


  /*Mixture model parameter between model1 and model2*/
    array[nParts] real<lower=0, upper=1> transfer_theta;                // Mixture proportion
    real<lower=0, upper=1> transfer_hier_theta_mu;         // Hierarchical smixture proportion

    transfer_hier_theta_mu = inv_logit(hier_theta_mu);				 
    for (p in 1:nParts) {
        transfer_theta[p] = inv_logit(hier_theta_mu + z_theta[p]*hier_theta_sd);
    }

}
model { 
    /* Hierarchical mu parameter for model 1*/    
    for (g in 1:nMeds_nSes){
        for (c in 1:nConds){
            hier_weight_mu[g,c] ~ normal(0,2);
            hier_alphaAct_pos_mu1[g,c] ~ normal(0,2);
            hier_alphaAct_neg_mu1[g,c] ~ normal(0,2);
            hier_alphaClr_pos_mu1[g,c] ~ normal(0,2);
            hier_alphaClr_neg_mu1[g,c] ~ normal(0,2);
            hier_sensitivity_mu[g,c] ~ normal(0,4);
        }
    }

    /* Hierarchical sd parameter foe model 1*/
    hier_alpha_sd1 ~ normal(0,1);  
    hier_weight_sd ~ normal(0,1); 
    hier_sensitivity_sd ~ normal(0,1);

    /* participant-level main paameter for model 1*/
    for (p in 1:nParts) {
        for (g in 1:nMeds_nSes){
            for (c in 1:nConds){
                z_weight[p, g, c] ~ normal(0,1);
                z_alphaAct_pos1[p, g, c] ~ normal(0,1);
                z_alphaAct_neg1[p, g, c] ~ normal(0,1);
                z_alphaClr_pos1[p, g, c] ~ normal(0,1);
                z_alphaClr_neg1[p, g, c] ~ normal(0,1);
                z_sensitivity[p, g, c] ~ normal(0,1); 
            }
        }
    }

    /* Hierarchical mu parameter for model 2*/    
    for (g in 1:nMeds_nSes){
        for (c in 1:nConds){
            hier_alphaAct_mu2[g,c] ~ normal(0,2);
            hier_alphaClr_mu2[g,c] ~ normal(0,2);
        }
    }

    /* Hierarchical sd parameter*/
    hier_alpha_sd2 ~ normal(0, 1);  
    
    /* participant-level main paameter*/
    for (p in 1:nParts) {
        for (g in 1:nMeds_nSes){
            for (c in 1:nConds){
                z_alphaAct2[p, g, c] ~ normal(0,1);
                z_alphaClr2[p, g, c] ~ normal(0,1);
            }
        }
    }

    /* Mixture model*/
    hier_theta_mu ~ normal(0,1);
    hier_theta_sd ~ normal(0,1);

    for (p in 1:nParts) {
        z_theta[p] ~ normal(0,1); 
    }

    /* RL likelihood */
    for (i in 1:N) { 
        vector[2] lps;
        lps[1] = log(transfer_theta[participant[i]]) + bernoulli_lpmf(pushed[i] | soft_max_EV1[i]);  // first term of mixture 
        lps[2] = log1m(transfer_theta[participant[i]]) + bernoulli_lpmf(pushed[i] | soft_max_EV2[i]);  // second term of mixture
        target += log_sum_exp(lps);
    }
}