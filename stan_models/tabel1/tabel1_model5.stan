data {
    int<lower=1> N;                                    // Number of trial-level observations
    int<lower=1> nParts;                               // Number of participants
    array[N] int<lower=0, upper=1> leftChosen;         // 1 if left and 0 if right 
    array[N] int<lower=0, upper=1> yellowChosen;       // 1 if yellow color is chosen and 0 if yellow color is not chosen 
    array[N] real<lower=0, upper=100> winAmtLeft;      // The amount of values feedback when left is correct response
    array[N] real<lower=0, upper=100> winAmtRight;     // The amount of values feedback when right is correct response
    array[N] real<lower=0, upper=100> winAmtYellow;    // The amount of values feedback when yellow chosen is correct response 
    array[N] real<lower=0, upper=100> winAmtBlue;      // The amount of values feedback when blue chosen is correct response 
    array[N] int<lower=0, upper=1> rewarded;           // 1 for rewarding and 0 for no-reward
    array[N] int<lower=1> participant;                 // Participant index for each trial
    array[N] int<lower=1> indicator;                   // Indicator of the first trial for each participant, run and conds 
    int<lower=1> n_conds_weight;                               // Number of cond, Action and Color value learning
    array[N] int<lower=1, upper=2> cond;          // 1 indicates first cond (Action) and 2 indicates second cond (Color)
} 
parameters {
    /* Hierarchical mu parameter*/                               
    real hier_alpha_pos_mu;               // Mean Hierarchical positive learning rate (unconstrained)
    real hier_alpha_neg_mu;               // Mean Hierarchical Negative Learning rate (unconstrained)
    array[n_conds_weight] real hier_weight_mu;    // Mean Hierarchical Weighting (unconstrained) 
    real hier_sensitivity_mu;             // Mean Hierarchical snesitivity (unconstrained)
    
    /* Hierarchical sd parameter*/                               
    real<lower=0> hier_alpha_sd;         // Between-participant variability Learning rate  
    real<lower=0> hier_weight_sd;        // Between-participant variability Wieghtening  
    real<lower=0> hier_sensitivity_sd;   // Between-participant variability sensitivity  

    /* participant-level main paameter*/
    array[nParts] real z_alpha_pos;        // Individual Positive Learning rate (unconstrained)
    array[nParts] real z_alpha_neg;        // Individual Negative Learning rate (unconstrained)
    array[nParts, n_conds_weight] real z_weight;   // Individual Wieghing (unconstrained)
    array[nParts] real z_sensitivity;      // Individual Sensitivity (unconstrained)  

}
transformed parameters {
    real p_left=0.5;           // Probability of reward for left responce
    real p_yell=0.5;           // Probability of reward for yrllow responce
    real EV_left=0;            // Standard Expected Value of left action
    real EV_right=0;           // Standard Expected Value of right action
    real EV_yell=0;            // Standard Expected Value of yellow action
    real EV_blue=0;            // Standard Expected Value of blue action
    real EV_left_yell=0;       // Weighting two strategies between left action and yellow color values learning
    real EV_left_blue=0;       // Weighting two strategies between left action and blue color values learning
    real EV_right_yell=0;      // Weighting two strategies between right action and yellow color values learning
    real EV_right_blue=0;      // Weighting two strategies between right action and blue color values learning
    vector[N] EV_diff=rep_vector(0,N);       // Expected value for each trial
   
    /* Transfer individual parameters */
    array[nParts] real<lower=0, upper=1> transfer_alpha_pos;        // Individual Poistive Learning rate (constrained) 
    array[nParts] real<lower=0, upper=1> transfer_alpha_neg;        // Individual Negative Learning rate (constrained)  
    array[nParts, n_conds_weight] real<lower=0, upper=1> transfer_weight;   // Individual Wieghing (constrained)  
    array[nParts] real<lower=0> transfer_sensitivity;               // Individual Sensitivity (constrained) 
    
    /* Transfer Hierarchical parameters just for output*/
    real<lower=0, upper=1> transfer_hier_alpha_pos_mu;              // Hierarchical Positive Learning rate (constrained)
    real<lower=0, upper=1> transfer_hier_alpha_neg_mu;              // Hierarchical Negative Learning rate (constrained)
    array[n_conds_weight] real<lower=0, upper=1> transfer_hier_weight_mu;   // Hierarchical Wieghting (constrained)
    real<lower=0> transfer_hier_sensitivity_mu;                     // Hierarchical snesitivity (constrained)

	transfer_hier_alpha_pos_mu = inv_logit(hier_alpha_pos_mu);				// for the output
	transfer_hier_alpha_neg_mu = inv_logit(hier_alpha_neg_mu);				 
    transfer_hier_weight_mu = inv_logit(hier_weight_mu);
	transfer_hier_sensitivity_mu = log1p_exp(hier_sensitivity_mu);

    for (p in 1:nParts) {
        for (c in 1:n_conds_weight){
            transfer_weight[p,c] = inv_logit(hier_weight_mu[c] + z_weight[p,c]*hier_weight_sd);
        }
        transfer_alpha_pos[p] = inv_logit(hier_alpha_pos_mu + z_alpha_pos[p]*hier_alpha_sd);
        transfer_alpha_neg[p] = inv_logit(hier_alpha_neg_mu + z_alpha_neg[p]*hier_alpha_sd);
        transfer_sensitivity[p] = log1p_exp(hier_sensitivity_mu + z_sensitivity[p]*hier_sensitivity_sd);
    }

    // Calculating the probability of reward
    for (i in 1:N) {
        // Restart probability of variable for each environemnt and cond
        if (indicator[i]==1){
            p_left = 0.5;
            p_yell = 0.5;
        }
        // Calculating the Standard Expected Value
        EV_left = p_left*winAmtLeft[i];
        EV_right = (1-p_left)*winAmtRight[i];
        EV_yell = p_yell*winAmtYellow[i];
        EV_blue = (1-p_yell)*winAmtBlue[i];
       
       // Relative contribution of ion Value Learning verus Color Value Learning
        EV_left_yell = transfer_weight[participant[i], cond[i]]*EV_left + (1 - transfer_weight[participant[i], cond[i]])*EV_yell;
        EV_left_blue = transfer_weight[participant[i], cond[i]]*EV_left + (1 - transfer_weight[participant[i], cond[i]])*EV_blue;
        EV_right_yell = transfer_weight[participant[i], cond[i]]*EV_right + (1 - transfer_weight[participant[i], cond[i]])*EV_yell;
        EV_right_blue = transfer_weight[participant[i], cond[i]]*EV_right + (1 - transfer_weight[participant[i], cond[i]])*EV_blue;
       
        // left/yellow coded and right/blue coded 1
        if ((leftChosen[i] == 1 && yellowChosen[i] == 1) || (leftChosen[i] == 0 && yellowChosen[i] == 0))
            EV_diff[i] = transfer_sensitivity[participant[i]] * (EV_left_yell - EV_right_blue);
            
        //  left/blue coded 1 and right/yellow coded 0
        else if ((leftChosen[i] == 1 && yellowChosen[i] == 0) || (leftChosen[i] == 0 && yellowChosen[i] == 1))
            EV_diff[i] = transfer_sensitivity[participant[i]] * (EV_left_blue - EV_right_yell);
            
        //RL rule update based on RPE in Action value learning
        if (leftChosen[i] == 1){
            // positive RPE
            if((rewarded[i] - p_left)>=0 ){ 
                p_left = p_left + transfer_alpha_pos[participant[i]]*(rewarded[i] - p_left);
            } 
            // negative RPE
            else{
                p_left = p_left + transfer_alpha_neg[participant[i]]*(rewarded[i] - p_left); 
            }
        }
        else{
            // positive RPE
            if((rewarded[i] + p_left - 1)>=0){ 
                p_left = p_left - transfer_alpha_pos[participant[i]]*(rewarded[i] + p_left - 1);
            } 
            // negative RPE
            else{
                p_left = p_left - transfer_alpha_neg[participant[i]]*(rewarded[i] + p_left - 1);
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
    for (c in 1:n_conds_weight){
        hier_weight_mu[c] ~ normal(0,2);
    }
    hier_alpha_pos_mu ~ normal(0,2);
    hier_alpha_neg_mu ~ normal(0,2);
    hier_sensitivity_mu ~ normal(0,3); 

    /* Hierarchical sd parameter*/
    hier_alpha_sd ~ normal(0,0.5);  
    hier_weight_sd ~ normal(0,0.5); 
    hier_sensitivity_sd ~ normal(0,0.5);
    
    /* participant-level main paameter*/
    for (p in 1:nParts) {
        for (c in 1:n_conds_weight){
            z_weight[p,c] ~ normal(0,1);
        }
        z_alpha_pos[p] ~ normal(0,1);
        z_alpha_neg[p] ~ normal(0,1);
        z_sensitivity[p] ~ normal(0,1); 
    }

    /* RL likelihood */
    for (i in 1:N) { 
        leftChosen[i] ~ bernoulli_logit(EV_diff[i]);
        }
}
generated quantities { 
   vector[N] log_lik;  
    /*  RL Log density likelihood */
    for (i in 1:N) {
        log_lik[i] = bernoulli_logit_lpmf(leftChosen[i] | EV_diff[i]);
    }
}