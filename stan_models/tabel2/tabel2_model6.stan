data {
    int<lower=1> N;                                    // Number of trial-level observations
    int<lower=1> nParts;                               // Number of participants
    array[N] int<lower=0, upper=1> pushed;             // 1 if pushed and 0 if pulled 
    array[N] int<lower=0, upper=1> yellowChosen;       // 1 if yellow color is chosen and 0 if blue color is chosen 
    array[N] real<lower=0, upper=100> winAmtPushable;  // The amount of values feedback when pushing is correct response
    array[N] real<lower=0, upper=100> winAmtPullable;  // The amount of values feedback when pulling is correct response
    array[N] real<lower=0, upper=100> winAmtYellow;    // The amount of values feedback when yellow chosen is correct response 
    array[N] real<lower=0, upper=100> winAmtBlue;      // The amount of values feedback when blue chosen is correct response 
    array[N] int<lower=0, upper=1> rewarded;           // 1 for rewarding and 0 for no-reward
    array[N] int<lower=1> participant;                 // Participant index for each trial
    array[N] int<lower=1> indicator;                   // Indicator of the first trial for each participant, run and conditions 
    int<lower=1> nConds;                               // Number of condition, Action and Color value learning
    array[N] int<lower=1, upper=2> condition;          // 1 indicates first condition (Action) and 2 indicates second condition (Color)
}
parameters {
    /* Hierarchical mu parameter*/                               
    real hier_alpha_pos_mu;                // Mean Hierarchical Positive Learning rate (unconstrained)
    real hier_alpha_neg_mu;                // Mean Hierarchical Negative Learning rate (unconstrained) 
    array[nConds] real hier_weight_mu;     // Mean Hierarchical Weighting (unconstrained) 
    real hier_sensitivity_mu;              // Mean Hierarchical snesitivity (unconstrained)
    
    /* Hierarchical sd parameter*/                               
    real<lower=0> hier_alpha_sd;           // Between-participant variability Learning rate
    real<lower=0> hier_weight_sd;          // Between-participant variability Wieghtening
    real<lower=0> hier_sensitivity_sd;     // Between-participant variability sensitivity

    /* participant-level main paameter*/
    array[nParts] real z_alpha_pos;        // Individual Positive Learning rate (unconstrained)
    array[nParts] real z_alpha_neg;        // Individual Negative Learning rate (unconstrained)
    array[nParts, nConds] real z_weight;   // Individual Wieghting (unconstrained)
    array[nParts] real z_sensitivity;      // Individual Sensitivity (unconstrained) 

}
transformed parameters {
    real p_push=.5;        // Probability of reward for pushing responce
    real p_yell=.5;        // Probability of reward for yrllow responce
    real EV_push=0;        // Standard Expected Value of push action
    real EV_pull=0;        // Standard Expected Value of pull action
    real EV_yell=0;        // Standard Expected Value of yellow action
    real EV_blue=0;        // Standard Expected Value of blue action
    real EV_push_yell=0;   // Weighting two strategies between push action and yellow color values learning
    real EV_push_blue=0;   // Weighting two strategies between push action and blue color values learning
    real EV_pull_yell=0;   // Weighting two strategies between pull action and yellow color values learning
    real EV_pull_blue=0;   // Weighting two strategies between pull action and blue color values learning
    vector[N] EV_diff=rep_vector(0,N);  // Expected value for each trial
   
    /* Transfer individual parameters */
    array[nParts] real<lower=0, upper=1> transfer_alpha_pos;       // Individual Poistive Learning rate (constrained)  
    array[nParts] real<lower=0, upper=1> transfer_alpha_neg;       // Individual Negative Learning rate (constrained) 
    array[nParts, nConds] real<lower=0, upper=1> transfer_weight;  // Individual Wieghting (constrained) 
    array[nParts] real<lower=0> transfer_sensitivity;              // Individual Sensitivity (constrained) 
    
    /* Transfer Hierarchical parameters just for output*/
    real<lower=0, upper=1> transfer_hier_alpha_pos_mu;             // Hierarchical Positive Learning rate (constrained) 
    real<lower=0, upper=1> transfer_hier_alpha_neg_mu;             // Hierarchical Negative Learning rate (constrained) 
    array[nConds] real<lower=0, upper=1> transfer_hier_weight_mu;  // Hierarchical Wieghtening (constrained) 
    real<lower=0> transfer_hier_sensitivity_mu;                    // Hierarchical snesitivity (constrained) 

	transfer_hier_alpha_pos_mu = inv_logit(hier_alpha_pos_mu);				// for the output
	transfer_hier_alpha_neg_mu = inv_logit(hier_alpha_neg_mu);				 
    transfer_hier_weight_mu = inv_logit(hier_weight_mu);
	transfer_hier_sensitivity_mu = log1p_exp(hier_sensitivity_mu);

    for (p in 1:nParts) {
        for (c in 1:nConds){
            transfer_weight[p,c] = inv_logit(hier_weight_mu[c] + z_weight[p,c]*hier_weight_sd);
        }
        transfer_alpha_pos[p] = inv_logit(hier_alpha_pos_mu + z_alpha_pos[p]*hier_alpha_sd);
        transfer_alpha_neg[p] = inv_logit(hier_alpha_neg_mu + z_alpha_neg[p]*hier_alpha_sd);
        transfer_sensitivity[p] = log1p_exp(hier_sensitivity_mu + z_sensitivity[p]*hier_sensitivity_sd);
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
        EV_push_yell = transfer_weight[participant[i], condition[i]]*EV_push + (1 - transfer_weight[participant[i], condition[i]])*EV_yell;
        EV_push_blue = transfer_weight[participant[i], condition[i]]*EV_push + (1 - transfer_weight[participant[i], condition[i]])*EV_blue;
        EV_pull_yell = transfer_weight[participant[i], condition[i]]*EV_pull + (1 - transfer_weight[participant[i], condition[i]])*EV_yell;
        EV_pull_blue = transfer_weight[participant[i], condition[i]]*EV_pull + (1 - transfer_weight[participant[i], condition[i]])*EV_blue;
       
        // pushed and yellow vs pulled and blue
        if ((pushed[i] == 1 && yellowChosen[i] == 1) || (pushed[i] == 0 && yellowChosen[i] == 0))
            EV_diff[i] =  transfer_sensitivity[participant[i]] * (EV_push_yell - EV_pull_blue);
            
        // pushed and blue vs pulled and yellow
        if ((pushed[i] == 1 && yellowChosen[i] == 0) || (pushed[i] == 0 && yellowChosen[i] == 1))
            EV_diff[i] = transfer_sensitivity[participant[i]] * (EV_push_blue - EV_pull_yell);
          
        //RL rule update based on RPE in Action value learning
        if (pushed[i] == 1){
            // positive RPE
            if((rewarded[i] - p_push)>=0 ){ 
                p_push = p_push + transfer_alpha_pos[participant[i]]*(rewarded[i] - p_push);
            } 
            // negative RPE
            else{
                p_push = p_push + transfer_alpha_neg[participant[i]]*(rewarded[i] - p_push); 
            }
        }
        else{
            // positive RPE
            if((rewarded[i] + p_push - 1)>=0){ 
                p_push = p_push - transfer_alpha_pos[participant[i]]*(rewarded[i] + p_push - 1);
            } 
            // negative RPE
            else{
                p_push = p_push - transfer_alpha_neg[participant[i]]*(rewarded[i] + p_push - 1);
            }
        }   

        //RL rule update based on RPE in Color value learning
        if (yellowChosen[i] == 1){
            // positive RPE
            if((rewarded[i] - p_yell)>=0){ 
                p_yell = p_yell + transfer_alpha_pos[participant[i]]*(rewarded[i] - p_yell);
            } 
            // negative RPE
            else{
                p_yell = p_yell + transfer_alpha_neg[participant[i]]*(rewarded[i] - p_yell);
            }
        }    
        else{
            // positive RPE
            if((rewarded[i] + p_yell - 1)>=0){ 
                p_yell = p_yell - transfer_alpha_pos[participant[i]]*(rewarded[i] + p_yell - 1);
            } 
            // negative RPE
            else{
                p_yell = p_yell - transfer_alpha_neg[participant[i]]*(rewarded[i] + p_yell - 1);
            }
        }
    }   
}
model { 
    /* Hierarchical mu parameter*/
    for (c in 1:nConds){
        hier_weight_mu[c] ~ normal(0,2);
    }
    hier_alpha_pos_mu ~ normal(0,2);
    hier_alpha_neg_mu ~ normal(0,2);
    hier_sensitivity_mu ~ normal(0,3); 

    /* Hierarchical sd parameter*/
    hier_alpha_sd ~ normal(0,.5);  
    hier_weight_sd ~ normal(0,.5); 
    hier_sensitivity_sd ~ normal(0,.5);
    
    /* participant-level main paameter*/
    for (p in 1:nParts) {
        for (c in 1:nConds){
            z_weight[p,c] ~ normal(0,1);
        }
        z_alpha_pos[p] ~ normal(0,1);
        z_alpha_neg[p] ~ normal(0,1);
        z_sensitivity[p] ~ normal(0,1); 
    }

    /* RL likelihood */
    for (i in 1:N) { 
        pushed[i] ~ bernoulli_logit(EV_diff[i]);
        }
}
generated quantities { 
   vector[N] log_lik;  
    /*  RL Log density likelihood */
    for (i in 1:N) {
        log_lik[i] = bernoulli_logit_lpmf(pushed[i] | EV_diff[i]);
    }
}