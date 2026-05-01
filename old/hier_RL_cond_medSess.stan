data {
    int<lower=1> N;                                     // Number of trial-level observations
    int<lower=1> nParts;                                // Number of participants
    array[N] int<lower=0, upper=1> pushed;              // 1 if pushed and 0 if pulled 
    array[N] int<lower=0, upper=1> yellowChosen;        // 1 if yellow color is chosen and 0 if yellow color is not chosen 
    array[N] real<lower=0, upper=100> winAmtPushable;   // The amount of values feedback when pushing is correct response
    array[N] real<lower=0, upper=100> winAmtPullable;   // The amount of values feedback when pulling is correct response
    array[N] real<lower=0, upper=100> winAmtYellow;     // The amount of values feedback when yellow chosen is correct response 
    array[N] real<lower=0, upper=100> winAmtBlue;       // The amount of values feedback when blue chosen is correct response 
    array[N] int<lower=0, upper=1> rewarded;            // 1 for rewarding and 0 for punishment
    array[N] int<lower=1> participant;                  // participant index for each trial
    array[N] int<lower=1> indicator;                    // indicator of the first trial of each participant, the first is denoted 1 otherwise 0
    int<lower=1> n_conds_alpha_pos;                     // Number of conditions for positive learning rate
    int<lower=1> n_medSess_alpha_pos;                   // Number of session/medication for positive learning rate
    int<lower=1> n_conds_alpha_neg;                     // Number of conditions for negative learning rate
    int<lower=1> n_medSess_alpha_neg;                   // Number of session/medication for negative learning rate
    int<lower=1> n_conds_weight;                        // Number of conditions for weighing
    int<lower=1> n_medSess_weight;                      // Number of session/medication for weighting 
    int<lower=1> n_conds_sensitivity;                   // Number of conditions for sensitivity
    int<lower=1> n_medSess_sensitivity;                 // Number of session/medication for sensitivity
    array[N] int<lower=1> cond;                         // Condition index per trial for positive learning rate
    array[N] int<lower=1> medSess;                      // medSess index per trial for positive learning rate
}
transformed data {
    /* chanage the conditions and medication/session array for each parameter*/
    array[N] int cond_alpha_pos;
    array[N] int cond_alpha_neg;
    array[N] int cond_weight;
    array[N] int cond_sensitivity;
    array[N] int medSess_alpha_pos;
    array[N] int medSess_alpha_neg;
    array[N] int medSess_weight;
    array[N] int medSess_sensitivity;

    for (i in 1:N) {
        cond_alpha_pos[i]    = (n_conds_alpha_pos == 2)    ? cond[i] : 1;
        cond_alpha_neg[i]    = (n_conds_alpha_neg == 2)    ? cond[i] : 1;
        cond_weight[i]       = (n_conds_weight == 2)       ? cond[i] : 1;
        cond_sensitivity[i]  = (n_conds_sensitivity == 2)  ? cond[i] : 1;
        medSess_alpha_pos[i] = (n_medSess_alpha_pos == 2)  ? medSess[i] : 1;
        medSess_alpha_neg[i] = (n_medSess_alpha_neg == 2)  ? medSess[i] : 1;
        medSess_weight[i]    = (n_medSess_weight == 2)     ? medSess[i] : 1;
        medSess_sensitivity[i] = (n_medSess_sensitivity == 2) ? medSess[i] : 1;
    }
}
parameters {
    /* Hierarchical mu parameters */                               
    array[n_conds_alpha_pos, n_medSess_alpha_pos] real hier_alpha_pos_mu;       // Mean Hierarchical Positive Learning rate (unconstrained)
    array[n_conds_alpha_neg, n_medSess_alpha_neg] real hier_alpha_neg_mu;       // Mean Hierarchical Negative Learning rate (unconstrained) 
    array[n_conds_weight, n_medSess_weight] real hier_weight_mu;                // Mean Hierarchical Weighting (unconstrained)
    array[n_conds_sensitivity, n_medSess_sensitivity] real hier_sensitivity_mu; // Mean Hierarchical sensitivity (unconstrained)
    
    /* Hierarchical sd parameters */                               
    real<lower=0> hier_alpha_sd;        // Between-participant variability learning rate
    real<lower=0> hier_weight_sd;       // Between-participant variability weighting
    real<lower=0> hier_sensitivity_sd;  // Between-participant variability sensitivity

    /* Participant-level parameters */
    array[nParts, n_conds_alpha_pos, n_medSess_alpha_pos] real z_alpha_pos;        // Individual positive learning rate (unconstrained)
    array[nParts, n_conds_alpha_neg, n_medSess_alpha_neg] real z_alpha_neg;        // Individual negative learning rate (unconstrained)
    array[nParts, n_conds_weight, n_medSess_weight] real z_weight;                 // Individual weighting parameter (unconstrained)  
    array[nParts, n_conds_sensitivity, n_medSess_sensitivity] real z_sensitivity;  // Individual sensitivity (unconstrained) 
}
transformed parameters {
    real p_push=.5;          // Probability of reward for pushing response
    real p_yell=.5;          // Probability of reward for yellow response
    real EV_push=0;         // Expected Value of push action
    real EV_pull=0;         // Expected Value of pull action
    real EV_yell=0;         // Expected Value of yellow action
    real EV_blue=0;         // Expected Value of blue action
    real EV_push_yell=0;    // Weighting between push action and yellow learning
    real EV_push_blue=0;    // Weighting between push action and blue learning
    real EV_pull_yell=0;    // Weighting between pull action and yellow learning
    real EV_pull_blue=0;    // Weighting between pull action and blue learning
    vector[N] EV_diff = rep_vector(0, N);  // Expected value difference for each trial
   
    /* Transfer individual parameters (constrained) */
    array[nParts, n_conds_alpha_pos, n_medSess_alpha_pos] real<lower=0, upper=1> transfer_alpha_pos;
    array[nParts, n_conds_alpha_neg, n_medSess_alpha_neg] real<lower=0, upper=1> transfer_alpha_neg;
    array[nParts, n_conds_weight, n_medSess_weight] real<lower=0, upper=1> transfer_weight;
    array[nParts, n_conds_sensitivity, n_medSess_sensitivity] real<lower=0> transfer_sensitivity;
    
    /* Transfer hierarchical parameters for output (constrained) */
    array[n_conds_alpha_pos, n_medSess_alpha_pos] real<lower=0, upper=1> transfer_hier_alpha_pos_mu;
    array[n_conds_alpha_neg, n_medSess_alpha_neg] real<lower=0, upper=1> transfer_hier_alpha_neg_mu;
    array[n_conds_weight, n_medSess_weight] real<lower=0, upper=1> transfer_hier_weight_mu;
    array[n_conds_sensitivity, n_medSess_sensitivity] real<lower=0> transfer_hier_sensitivity_mu;

	transfer_hier_alpha_pos_mu = inv_logit(hier_alpha_pos_mu);				// for the output positive
	transfer_hier_alpha_neg_mu = inv_logit(hier_alpha_neg_mu);				 
    transfer_hier_weight_mu = inv_logit(hier_weight_mu);
    for (c in 1:n_conds_sensitivity)
        for (s in 1:n_medSess_sensitivity)
            transfer_hier_sensitivity_mu[c,s] = log1p_exp(hier_sensitivity_mu[c,s]);

    // Individual parameter transformations
    for (p in 1:nParts) {
        for (c in 1:n_conds_alpha_pos)                              // Individual positive learning rate (constrained)
            for (s in 1:n_medSess_alpha_pos)
                transfer_alpha_pos[p,c,s] = inv_logit(hier_alpha_pos_mu[c,s] + z_alpha_pos[p,c,s] * hier_alpha_sd);
        for (c in 1:n_conds_alpha_neg)                             // Individual negative learning rate (constrained)
            for (s in 1:n_medSess_alpha_neg)                       
                transfer_alpha_neg[p,c,s] = inv_logit(hier_alpha_neg_mu[c,s] + z_alpha_neg[p,c,s] * hier_alpha_sd);
        for (c in 1:n_conds_weight)                                // Individual weighting (constrained)
            for (s in 1:n_medSess_weight)             
                transfer_weight[p,c,s] = inv_logit(hier_weight_mu[c,s] + z_weight[p,c,s] * hier_weight_sd);
        for (c in 1:n_conds_sensitivity)                           // Individual weighting (constrained) 
            for (s in 1:n_medSess_sensitivity)
                transfer_sensitivity[p,c,s] = log1p_exp(hier_sensitivity_mu[c,s] + z_sensitivity[p,c,s] * hier_sensitivity_sd);
    }
   
    // Trial loop: compute EV and update RL probabilities
    for (n in 1:N) {

        // Restart probability for each new participant block
        if (indicator[n] == 1) {
            p_push = 0.5;
            p_yell = 0.5;
        }

        // Standard Expected Values
        EV_push = p_push * winAmtPushable[n];
        EV_pull = (1 - p_push) * winAmtPullable[n];
        EV_yell = p_yell * winAmtYellow[n];
        EV_blue = (1 - p_yell) * winAmtBlue[n];
       
        // Weighted combination of action and color EVs
        EV_push_yell = transfer_weight[participant[n], cond_weight[n], medSess_weight[n]] * EV_push + (1 - transfer_weight[participant[n], cond_weight[n], medSess_weight[n]]) * EV_yell;
        EV_push_blue = transfer_weight[participant[n], cond_weight[n], medSess_weight[n]] * EV_push + (1 - transfer_weight[participant[n], cond_weight[n], medSess_weight[n]]) * EV_blue;
        EV_pull_yell = transfer_weight[participant[n], cond_weight[n], medSess_weight[n]] * EV_pull + (1 - transfer_weight[participant[n], cond_weight[n], medSess_weight[n]]) * EV_yell;
        EV_pull_blue = transfer_weight[participant[n], cond_weight[n], medSess_weight[n]] * EV_pull + (1 - transfer_weight[participant[n], cond_weight[n], medSess_weight[n]]) * EV_blue;
       
        // Softmax input: EV difference scaled by sensitivity
        if ((pushed[n] == 1 && yellowChosen[n] == 1) || (pushed[n] == 0 && yellowChosen[n] == 0)) // pushed/yellow coded 1 and pulled/blue coded 0
            EV_diff[n] = transfer_sensitivity[participant[n], cond_sensitivity[n], medSess_sensitivity[n]] * (EV_push_yell - EV_pull_blue);
        else                                                                                      //  pushed/blue coded 1 and pulled/yellow coded 0
            EV_diff[n] = transfer_sensitivity[participant[n], cond_sensitivity[n], medSess_sensitivity[n]] * (EV_push_blue - EV_pull_yell);

        // RL update: Action value learning
        if (pushed[n] == 1) {                  // pushed
            if ((rewarded[n] - p_push) >= 0)  // positive RPE
                p_push = p_push + transfer_alpha_pos[participant[n], cond_alpha_pos[n], medSess_alpha_pos[n]] * (rewarded[n] - p_push);
            else                               // positive RPE
                p_push = p_push + transfer_alpha_neg[participant[n], cond_alpha_neg[n], medSess_alpha_neg[n]] * (rewarded[n] - p_push);
        } else {                               // puslled
            if ((rewarded[n] + p_push - 1) >= 0)
                p_push = p_push - transfer_alpha_pos[participant[n], cond_alpha_pos[n], medSess_alpha_pos[n]] * (rewarded[n] + p_push - 1);
            else
                p_push = p_push - transfer_alpha_neg[participant[n], cond_alpha_neg[n], medSess_alpha_neg[n]] * (rewarded[n] + p_push - 1);
        }   

        // RL update: Color value learning
        if (yellowChosen[n] == 1) {            // yellow is chosen
            if ((rewarded[n] - p_yell) >= 0)  // positive RPE
                p_yell = p_yell + transfer_alpha_pos[participant[n], cond_alpha_pos[n], medSess_alpha_pos[n]] * (rewarded[n] - p_yell);
            else                              // Negative RPE
                p_yell = p_yell + transfer_alpha_neg[participant[n], cond_alpha_neg[n], medSess_alpha_neg[n]] * (rewarded[n] - p_yell);
        } else {                              // blue is chosen
            if ((rewarded[n] + p_yell - 1) >= 0)
                p_yell = p_yell - transfer_alpha_pos[participant[n], cond_alpha_pos[n], medSess_alpha_pos[n]] * (rewarded[n] + p_yell - 1);
            else
                p_yell = p_yell - transfer_alpha_neg[participant[n], cond_alpha_neg[n], medSess_alpha_neg[n]] * (rewarded[n] + p_yell - 1);
        }
    }   
}
model { 
    /* Hierarchical mu priors */
    for (c in 1:n_conds_alpha_pos)                    // Hierarchical positive learning rate (unconstrained)
        for (s in 1:n_medSess_alpha_pos)
            hier_alpha_pos_mu[c,s] ~ normal(0, 2);

    for (c in 1:n_conds_alpha_neg)                     // Mean Hierarchical negative learning rate (unconstrained)
        for (s in 1:n_medSess_alpha_neg)
            hier_alpha_neg_mu[c,s] ~ normal(0, 2);

    for (c in 1:n_conds_weight)                         // Mean Hierarchical weighing (unconstrained)
        for (s in 1:n_medSess_weight) 
            hier_weight_mu[c,s] ~ normal(0, 2);

    for (c in 1:n_conds_sensitivity)                    // Mean Hierarchical sensitivity (unconstrained)
        for (s in 1:n_medSess_sensitivity)
            hier_sensitivity_mu[c,s] ~ normal(0, 3);

    /* Hierarchical sd priors */
    hier_alpha_sd ~ normal(0, 1) T[0,];  
    hier_weight_sd ~ normal(0, 1) T[0,]; 
    hier_sensitivity_sd ~ normal(0, 1) T[0,];
    
    /* Participant-level priors */
    for (p in 1:nParts) {
        for (c in 1:n_conds_alpha_pos)                 // Individual positive learning rate (unconstrained)    
            for (s in 1:n_medSess_alpha_pos)
                z_alpha_pos[p,c,s] ~ normal(0, 1);
       
        for (c in 1:n_conds_alpha_neg)                // Individual negative learning rate (unconstrained)
            for (s in 1:n_medSess_alpha_neg) 
                z_alpha_neg[p,c,s] ~ normal(0, 1);

        for (c in 1:n_conds_weight)                      // Individual weighting (unconstrained)
            for (s in 1:n_medSess_weight)
                z_weight[p,c,s] ~ normal(0, 1);

        for (c in 1:n_conds_sensitivity)                 // Individual weighting (unconstrained)
            for (s in 1:n_medSess_sensitivity)
                z_sensitivity[p,c,s] ~ normal(0, 1);
    }

    /* RL likelihood */
    for (n in 1:N)
        pushed[n] ~ bernoulli_logit(EV_diff[n]);
}
generated quantities { 
    vector[N] log_lik;  
    /*  RL Log density likelihood */
    for (n in 1:N)
        log_lik[n] = bernoulli_logit_lpmf(pushed[n] | EV_diff[n]);


    /* Individual positive learning rate (unconstrained) */
    array[nParts, n_conds_weight, n_medSess_weight] real weight;
    for (p in 1:nParts)
        for (c in 1:n_conds_weight)
            for (s in 1:n_medSess_weight)
                weight[p,c,s] = hier_weight_mu[c,s] + z_weight[p,c,s] * hier_weight_sd;

}
