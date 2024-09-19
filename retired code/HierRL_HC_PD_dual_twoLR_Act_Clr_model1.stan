data {
    // Healthy Control
    int<lower=1> N_HC;        // Number of trial-level observations
    int<lower=1> nParts_HC;   // Number of participants
    array[N_HC] int<lower=0, upper=1> pushed_HC;             // 1 if pushed and 0 if pulled 
    array[N_HC] int<lower=0, upper=1> yellowChosen_HC;       // 1 if yellow color is chosen and 0 if yellow color is not chosen 
    array[N_HC] real<lower=0, upper=100> winAmtPushable_HC;  // The amount of values feedback when pushing is correct response
    array[N_HC] real<lower=0, upper=100> winAmtPullable_HC;  // The amount of values feedback when pulling is correct response
    array[N_HC] real<lower=0, upper=100> winAmtYellow_HC;    // The amount of values feedback when yellow chosen is correct response 
    array[N_HC] real<lower=0, upper=100> winAmtBlue_HC;      // The amount of values feedback when blue chosen is correct response 
    array[N_HC] int<lower=0, upper=1> rewarded_HC;           // 1 for rewarding and 0 for punishment
    array[N_HC] int<lower=1> participant_HC;                 // participant index for each trial
    array[N_HC] int<lower=1> indicator_HC;                   // indicator of the first trial of each participant, the first is denoted 1 otherwise 0
    int<lower=1> nMeds_nSes_HC;                         // Number of medication_session level (OFF vs ON)
    int<lower=1> nConds_HC;                        // Number of condition, Action and Color value learning
    array[N_HC] int<lower=1, upper=2> medication_session_HC;       // 1 indecates OFF medication_session and 2 indicates On medication_session
    array[N_HC] int<lower=1, upper=2> condition_HC;   // 1 indecates first codition (Action) and 2 indicates second condition (Color)
    real<lower=0, upper=1> p_push_init_HC;         // Initial value of reward probability for pushed responce
    real<lower=0, upper=1> p_yell_init_HC;         // Initial value of reward probability for Color responce
  
}
parameters {
    // for Healthy Control
    /* Hierarchical mu parameter*/                               
    array[nMeds_nSes_HC, nConds_HC] real hier_alphaAct_pos_mu_HC;   // Mean Hierarchical Positive Learning rate for action Learning Value and medication_session effect
    array[nMeds_nSes_HC, nConds_HC] real hier_alphaAct_neg_mu_HC;   // Mean Hierarchical Negative Learning rate for action Learning Value and medication_session effect
    array[nMeds_nSes_HC, nConds_HC] real hier_alphaClr_pos_mu_HC;   // Mean Hierarchical Positive Learning rate for color Learning Value and medication_session effect
    array[nMeds_nSes_HC, nConds_HC] real hier_alphaClr_neg_mu_HC;   // Mean Hierarchical Negative Learning rate for color Learning Value and medication_session effect
    array[nMeds_nSes_HC, nConds_HC] real hier_weight_mu_HC;         // Mean Hierarchical Wieghtening of Action Learning Value against to Color Learnig Value
    array[nMeds_nSes_HC, nConds_HC] real hier_sensitivity_mu_HC;    // Mean Hierarchical snesitivity, With a higher sensitivity value θ, choices are more sensitive to value differences
  
    /* Hierarchical sd parameter*/                               
    real<lower=0> hier_alpha_sd_HC;       // Between-participant variability Learning rate for Learning Value
    real<lower=0> hier_weight_sd_HC;      // Between-participant variability Wieghtening of Action Learning Value against to Color Learnig Value
    real<lower=0> hier_sensitivity_sd_HC; // Between-participant variability sensitivity

    /* participant-level main paameter*/
    array[nParts_HC, nMeds_nSes_HC, nConds_HC] real z_alphaAct_pos_HC;   // Positive Learning rate for Action Learning Value
    array[nParts_HC, nMeds_nSes_HC, nConds_HC] real z_alphaAct_neg_HC;   // Negative Learning rate for Action Learning Value
    array[nParts_HC, nMeds_nSes_HC, nConds_HC] real z_alphaClr_pos_HC;   // Positive Learning rate for Color Learning Value
    array[nParts_HC, nMeds_nSes_HC, nConds_HC] real z_alphaClr_neg_HC;   // Negative Learning rate for Color Learning Value
    array[nParts_HC, nMeds_nSes_HC, nConds_HC] real z_weight_HC;  // Wieghtening of Action Learning Value against to Learnig Value
    array[nParts_HC, nMeds_nSes_HC, nConds_HC] real z_sensitivity_HC;         // With a higher sensitivity value θ, choices are more sensitive to value differences

}
transformed parameters {
    // Healthy control
    /* probability of each features and their combination*/
    real p_push_HC;   // Probability of reward for pushing responce
    real p_pull_HC;   // Probability of reward for pulling responce
    real p_yell_HC;   // Probability of reward for yrllow responce
    real p_blue_HC;   // Probability of reward for blue responce
    real EV_push_HC;  // Standard Expected Value of push action
    real EV_pull_HC;  // Standard Expected Value of pull action
    real EV_yell_HC;  // Standard Expected Value of yellow action
    real EV_blue_HC;  // Standard Expected Value of blue action
    real EV_push_yell_HC;      // Weighting two strategies between push action and yellow color values learning
    real EV_push_blue_HC;      // Weighting two strategies between push action and blue color values learning
    real EV_pull_yell_HC;      // Weighting two strategies between pull action and yellow color values learning
    real EV_pull_blue_HC;      // Weighting two strategies between pull action and blue color values learning
    vector[N_HC] soft_max_EV_HC;  //  The soft-max function for each trial, trial-by-trial probability
   
    /* Transfer individual parameters */
    array[nParts_HC, nMeds_nSes_HC, nConds_HC] real<lower=0, upper=1> transfer_alphaAct_pos_HC;   // Poistive Learning rate for Action Learning Value
    array[nParts_HC, nMeds_nSes_HC, nConds_HC] real<lower=0, upper=1> transfer_alphaAct_neg_HC;   // Negative Learning rate for Action Learning Value
    array[nParts_HC, nMeds_nSes_HC, nConds_HC] real<lower=0, upper=1> transfer_alphaClr_pos_HC;   // Positive Learning rate for Color Learning Value
    array[nParts_HC, nMeds_nSes_HC, nConds_HC] real<lower=0, upper=1> transfer_alphaClr_neg_HC;   // Negative Learning rate for Color Learning Value
    array[nParts_HC, nMeds_nSes_HC, nConds_HC] real<lower=0, upper=1> transfer_weight_HC;  // Wieghtening of Action Learning Value against to Color Learnig Value
    array[nParts_HC, nMeds_nSes_HC, nConds_HC] real<lower=0> transfer_sensitivity_HC;         // With a higher sensitivity value θ, choices are more sensitive to value differences
    
    /* Transfer Hierarchical parameters just for output*/
    array[nMeds_nSes_HC, nConds_HC] real<lower=0, upper=1> transfer_hier_alphaAct_pos_mu_HC;   // Hierarchical Positive Learning rate for Action Learning Value
    array[nMeds_nSes_HC, nConds_HC] real<lower=0, upper=1> transfer_hier_alphaAct_neg_mu_HC;   // Hierarchical Negative Learning rate for Action Learning Value
    array[nMeds_nSes_HC, nConds_HC] real<lower=0, upper=1> transfer_hier_alphaClr_pos_mu_HC;   // Hierarchical Positive Learning rate for Color Learning Value
    array[nMeds_nSes_HC, nConds_HC] real<lower=0, upper=1> transfer_hier_alphaClr_neg_mu_HC;   // Hierarchical Negative  Learning rate for Color Learning Value
    array[nMeds_nSes_HC, nConds_HC] real<lower=0, upper=1> transfer_hier_weight_mu_HC;  // Hierarchical Wieghtening of Action Learning Value against to Color Learnig Value
    array[nMeds_nSes_HC, nConds_HC] real<lower=0> transfer_hier_sensitivity_mu_HC;         // Hierarchical snesitivity, With a higher sensitivity value θ, choices are more sensitive to value differences

	transfer_hier_alphaAct_pos_mu_HC = Phi(hier_alphaAct_pos_mu_HC);				// for the output
	transfer_hier_alphaAct_neg_mu_HC = Phi(hier_alphaAct_neg_mu_HC);				 
	transfer_hier_alphaClr_pos_mu_HC = Phi(hier_alphaClr_pos_mu_HC);				 
	transfer_hier_alphaClr_neg_mu_HC = Phi(hier_alphaClr_neg_mu_HC);				 
    transfer_hier_weight_mu_HC = Phi(hier_weight_mu_HC);
	for (g in 1:nMeds_nSes_HC){
        for (c in 1:nConds_HC){
            transfer_hier_sensitivity_mu_HC[g, c] = log(1 + exp(hier_sensitivity_mu_HC[g, c]));
        }
    }

    for (p in 1:nParts_HC) {
        for (g in 1:nMeds_nSes_HC){
            for (c in 1:nConds_HC){
                transfer_weight_HC[p, g, c] = Phi(hier_weight_mu_HC[g, c] + z_weight_HC[p, g, c]*hier_weight_sd_HC);
                transfer_alphaAct_pos_HC[p, g, c] = Phi(hier_alphaAct_pos_mu_HC[g, c] + z_alphaAct_pos_HC[p, g, c]*hier_alpha_sd_HC);
                transfer_alphaAct_neg_HC[p, g, c] = Phi(hier_alphaAct_neg_mu_HC[g, c] + z_alphaAct_neg_HC[p, g, c]*hier_alpha_sd_HC);
                transfer_alphaClr_pos_HC[p, g, c] = Phi(hier_alphaClr_pos_mu_HC[g, c] + z_alphaClr_pos_HC[p, g, c]*hier_alpha_sd_HC);
                transfer_alphaClr_neg_HC[p, g, c] = Phi(hier_alphaClr_neg_mu_HC[g, c] + z_alphaClr_neg_HC[p, g, c]*hier_alpha_sd_HC);
                transfer_sensitivity_HC[p, g, c] = log(1 + exp(hier_sensitivity_mu_HC[g, c] + z_sensitivity_HC[p,g, c]*hier_sensitivity_sd_HC));
            }
        }
    }

    // Calculating the probability of reward
   for (i in 1:N_HC) {
        // Restart probability of variable for each environemnt and condition
        if (indicator_HC[i]==1){
            p_push_HC = p_push_init_HC;
            p_pull_HC = 1 - p_push_init_HC;
            p_yell_HC = p_yell_init_HC;
            p_blue_HC = 1 - p_yell_init_HC;
        }
        // Calculating the Standard Expected Value
        EV_push_HC = p_push_HC*winAmtPushable_HC[i];
        EV_pull_HC = p_pull_HC*winAmtPullable_HC[i];
        EV_yell_HC = p_yell_HC*winAmtYellow_HC[i];
        EV_blue_HC = p_blue_HC*winAmtBlue_HC[i];
       
        // Relative contribution of Action Value Learning verus Color Value Learning
        EV_push_yell_HC = transfer_weight_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]]*EV_push_HC + (1 - transfer_weight_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]])*EV_yell_HC;
        EV_push_blue_HC = transfer_weight_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]]*EV_push_HC + (1 - transfer_weight_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]])*EV_blue_HC;
        EV_pull_yell_HC = transfer_weight_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]]*EV_pull_HC + (1 - transfer_weight_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]])*EV_yell_HC;
        EV_pull_blue_HC = transfer_weight_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]]*EV_pull_HC + (1 - transfer_weight_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]])*EV_blue_HC;
       
        /* Calculating the soft-max function over weightening Action and Color conditions*/ 
        // pushed/yellow coded and pulled/blue coded 1
        if ((pushed_HC[i] == 1 && yellowChosen_HC[i] == 1) || (pushed_HC[i] == 0 && yellowChosen_HC[i] == 0))
            soft_max_EV_HC[i] = exp(transfer_sensitivity_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]]*EV_push_yell_HC)/(exp(transfer_sensitivity_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]]*EV_push_yell_HC) + exp(transfer_sensitivity_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]]*EV_pull_blue_HC));

        //  pushed/blue coded 1 and pulled/yellow coded 0
        if ((pushed_HC[i] == 1 && yellowChosen_HC[i] == 0) || (pushed_HC[i] == 0 && yellowChosen_HC[i] == 1))
            soft_max_EV_HC[i] = exp(transfer_sensitivity_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]]*EV_push_blue_HC)/(exp(transfer_sensitivity_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]]*EV_push_blue_HC) + exp(transfer_sensitivity_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]]*EV_pull_yell_HC));  
          
        // RL rule update for computing prediction error and internal value expectation for the next trial based on the current reward output and interal value expectation
        /*Action value learning*/
        if (pushed_HC[i] == 1){
            // positive RPE
            if((rewarded_HC[i] - p_push_HC )>=0 ){ 
                p_push_HC = p_push_HC + transfer_alphaAct_pos_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]]*(rewarded_HC[i] - p_push_HC);
            } 
            // negative RPE
            else{
                p_push_HC = p_push_HC + transfer_alphaAct_neg_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]]*(rewarded_HC[i] - p_push_HC); 
            }
        }
        else{
            // positive RPE
            if((rewarded_HC[i] - p_push_HC )>=0 ){ 
                p_pull_HC = p_pull_HC + transfer_alphaAct_pos_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]]*(rewarded_HC[i] - p_pull_HC);
            } 
            // negative RPE
            else{
                p_pull_HC = p_pull_HC + transfer_alphaAct_neg_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]]*(rewarded_HC[i] - p_pull_HC);
            }
        }   

        /*Color value learning*/
        if (yellowChosen_HC[i] == 1){
            // positive RPE
            if((rewarded_HC[i] - p_push_HC )>=0 ){ 
                p_yell_HC = p_yell_HC + transfer_alphaClr_pos_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]]*(rewarded_HC[i] - p_yell_HC);
            } 
            // negative RPE
            else{
                p_yell_HC = p_yell_HC + transfer_alphaClr_neg_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]]*(rewarded_HC[i] - p_yell_HC);
            }
        }    
        else{
            // positive RPE
            if((rewarded_HC[i] - p_push_HC )>=0 ){ 
                p_blue_HC = p_blue_HC + transfer_alphaClr_pos_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]]*(rewarded_HC[i] - p_blue_HC);
            } 
            // negative RPE
            else{
                p_blue_HC = p_blue_HC + transfer_alphaClr_neg_HC[participant_HC[i], medication_session_HC[i], condition_HC[i]]*(rewarded_HC[i] - p_blue_HC);
            }
        }
    }   
}
model { 
    /* Hierarchical mu parameter*/    
        for (g in 1:nMeds_nSes_HC){
            for (c in 1:nConds_HC){
                hier_weight_mu_HC[g,c] ~ normal(0,1);
                hier_alphaAct_pos_mu_HC[g,c] ~ normal(0,1);
                hier_alphaAct_neg_mu_HC[g,c] ~ normal(0,1);
                hier_alphaClr_pos_mu_HC[g,c] ~ normal(0,1);
                hier_alphaClr_neg_mu_HC[g,c] ~ normal(0,1);
                hier_sensitivity_mu_HC[g,c] ~ normal(1,5); 
            }
        }

    /* Hierarchical sd parameter*/
    hier_alpha_sd_HC ~ normal(0,.1) T[0,];  
    hier_weight_sd_HC ~ normal(0,.1) T[0,]; 
    hier_sensitivity_sd_HC ~ normal(0,.1) T[0,];
    
    /* participant-level main paameter*/
    for (p in 1:nParts_HC) {
        for (g in 1:nMeds_nSes_HC){
            for (c in 1:nConds_HC){
                z_weight_HC[p, g, c] ~ normal(0,1);
                z_alphaAct_pos_HC[p, g, c] ~ normal(0,1);
                z_alphaAct_neg_HC[p, g, c] ~ normal(0,1);
                z_alphaClr_pos_HC[p, g, c] ~ normal(0,1);
                z_alphaClr_neg_HC[p, g, c] ~ normal(0,1);
                z_sensitivity_HC[p, g, c] ~ normal(0,1); 
            }
        }
    }

    /* RL likelihood */
    for (i in 1:N_HC) { 
        pushed_HC[i] ~ bernoulli(soft_max_EV_HC[i]);
        }
} 