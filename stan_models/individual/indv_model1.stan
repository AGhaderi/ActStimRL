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
    int<lower=1> nMeds_nSes;                           // Number of medication_session level (OFF vs ON)
    array[N] int<lower=1, upper=2> medication_session; // 1 indecates OFF medication_session and 2 indicates On medication_session

}
parameters {
    /* participant-level main paameter*/
    array[nParts, nConds, nMeds_nSes] real alpha_pos;    // Positive Learning rate
    array[nParts, nConds, nMeds_nSes] real alpha_neg;    // Negative Learning rate
    array[nParts, nConds, nMeds_nSes] real weight;       // Wieghtening
    array[nParts, nConds, nMeds_nSes] real sensitivity;  // Sensitivity  
}
transformed parameters {
    real p_push=.5;          // Probability of reward for pushing responce
    real p_yell=.5;          // Probability of reward for yrllow responce
    real EV_push=0;          // Standard Expected Value of push action
    real EV_pull=0;          // Standard Expected Value of pull action
    real EV_yell=0;          // Standard Expected Value of yellow action
    real EV_blue=0;          // Standard Expected Value of blue action
    real EV_push_yell=0;     // Weighting two strategies between push action and yellow color values learning
    real EV_push_blue=0;     // Weighting two strategies between push action and blue color values learning
    real EV_pull_yell=0;     // Weighting two strategies between pull action and yellow color values learning
    real EV_pull_blue=0;     // Weighting two strategies between pull action and blue color values learning
    vector[N] EV_diff=rep_vector(0,N);     // Expected value for each trial

    /* Transfer individual parameters */
    array[nParts, nConds, nMeds_nSes] real<lower=0, upper=1> transfer_alpha_pos;   // Poistive Learning rate  
    array[nParts, nConds, nMeds_nSes] real<lower=0, upper=1> transfer_alpha_neg;   // Negative Learning rate  
    array[nParts, nConds, nMeds_nSes] real<lower=0, upper=1> transfer_weight;      // Wieghtening  
    array[nParts, nConds, nMeds_nSes] real<lower=0> transfer_sensitivity;          // Sensitivity 
    
    for (p in 1:nParts) {
        for (c in 1:nConds){
            for (s in 1:nMeds_nSes){
                transfer_alpha_pos[p,c,s] = inv_logit(alpha_pos[p,c,s]);
                transfer_alpha_neg[p,c,s] = inv_logit(alpha_neg[p,c,s]);
                transfer_weight[p,c,s] = inv_logit(weight[p,c,s]);
                transfer_sensitivity[p,c,s] = log1p_exp(sensitivity[p,c,s]);
            }   
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
            EV_diff[i] = transfer_sensitivity[participant[i], condition[i], medication_session[i]] * (EV_push_yell - EV_pull_blue);

        //  pushed/blue coded 1 and pulled/yellow coded 0
        else if ((pushed[i] == 1 && yellowChosen[i] == 0) || (pushed[i] == 0 && yellowChosen[i] == 1))
            EV_diff[i] = transfer_sensitivity[participant[i], condition[i], medication_session[i]] * (EV_push_blue - EV_pull_yell);

        //RL rule update based on RPE in Action value learning
        if (pushed[i] == 1){
            // positive RPE
            if((rewarded[i] - p_push)>=0 ){ 
                p_push = p_push + transfer_alpha_pos[participant[i], condition[i], medication_session[i]]*(rewarded[i] - p_push);
            } 
            // negative RPE
            else{
                p_push = p_push + transfer_alpha_neg[participant[i], condition[i], medication_session[i]]*(rewarded[i] - p_push); 
            }
        }
        else{
            // positive RPE
            if((rewarded[i] + p_push - 1)>=0){ 
                p_push = p_push - transfer_alpha_pos[participant[i], condition[i], medication_session[i]]*(rewarded[i] + p_push - 1);
            } 
            // negative RPE
            else{
                p_push = p_push - transfer_alpha_neg[participant[i], condition[i], medication_session[i]]*(rewarded[i] + p_push - 1);
            }
        }   

        //RL rule update based on RPE in Color value learning
        if (yellowChosen[i] == 1){
            // positive RPE
            if((rewarded[i] - p_yell)>=0){ 
                p_yell = p_yell + transfer_alpha_pos[participant[i], condition[i], medication_session[i]]*(rewarded[i] - p_yell);
            } 
            // negative RPE
            else{
                p_yell = p_yell + transfer_alpha_neg[participant[i], condition[i], medication_session[i]]*(rewarded[i] - p_yell);
            }
        }    
        else{
            // positive RPE
            if((rewarded[i] + p_yell - 1)>=0){ 
                p_yell = p_yell - transfer_alpha_pos[participant[i], condition[i], medication_session[i]]*(rewarded[i] + p_yell - 1);
            } 
            // negative RPE
            else{
                p_yell = p_yell - transfer_alpha_neg[participant[i], condition[i], medication_session[i]]*(rewarded[i] + p_yell - 1);
            }
        }
    }   
}
model {       
    /* participant-level main paameter*/
    for (p in 1:nParts) {
        for (c in 1:nConds){
            for (s in 1:nMeds_nSes){
                alpha_pos[p,s] ~ normal(0,3);
                alpha_neg[p,c,s] ~ normal(0,3);
                weight[p,c,s] ~ normal(0,3);
                sensitivity[p,c,s] ~ normal(0,5); 
            }
        }
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