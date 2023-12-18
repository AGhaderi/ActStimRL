/* Model RL in addditon to weightening parameter to modele both Action and Color values learning at the same time
    This code can only model each singel data for instance one condition in one specific session and run.
    This code conducts the Hierarchical level nanalysis
    For both distinct options the push is coded 1 if selected and 0 if not selected. But for the first option, yell chosen encoded 1 and blue is conded 0
    on the other hand, in the second option blue is encoded 1 and yellow is encoded 0. At the result, since the push matches to yellow in the first option
    and maches the blue in the second option, there is no necessary to change anything, we should just consider the push encoding.
*/ 

data {
    int<lower=1> N;        // Number of trial-level observations
    int<lower=1> nParts;   // Number of participants
    int<lower=0, upper=1> pushed[N];            // 1 if pushed and 0 if pulled 
    int<lower=0, upper=1> yellowChosen[N];      // 1 if yellow color is chosen and 0 if yellow color is not chosen 
    real<lower=0, upper=100> winAmtPushable[N]; // The amount of values feedback when pushing is correct response
    real<lower=0, upper=100> winAmtPullable[N]; // The amount of values feedback when pulling is correct response
    real<lower=0, upper=100> winAmtYellow[N];   // The amount of values feedback when yellow chosen is correct response 
    real<lower=0, upper=100> winAmtBlue[N];     // The amount of values feedback when blue chosen is correct response 
    int<lower=0, upper=1> rewarded[N];          // 1 for rewarding and 0 for punishment
    int<lower=1> participant[N];         // participant index for each trial
    int<lower=1> indicator[N];           // indicator of the first trial of each participant, the first is denoted 1 otherwise 0
    real<lower=0, upper=1> p_push_init;  // Initial value of reward probability for pushed responce
    real<lower=0, upper=1> p_yell_init;  // Initial value of reward probability for Color responce
}
parameters {
    /* Hierarchical mu parameter*/                               
    real<lower=0, upper=1> alphaAct_hier;   // Hierarchical Learning rate for Action Learning Value
    real<lower=0, upper=1> alphaClr_hier;   // Hierarchical Learning rate for Color Learning Value
    real<lower=0, upper=1> weightAct_hier;  // Hierarchical Wieghtening of Action Learning Value against to Color Learnig Value
    real<lower=0> sensitivity_hier;         // Hierarchical snesitivity, With a higher sensitivity value θ, choices are more sensitive to value differences

    /* Hierarchical sd parameter*/                               
    real<lower=0> alphaAct_sd;      // Between-participant variability Learning rate for Action Learning Value
    real<lower=0> alphaClr_sd;      // Between-participant variability Learning rate for Color Learning Value
    real<lower=0> weightAct_sd;     // Between-participant variability Wieghtening of Action Learning Value against to Color Learnig Value
    real<lower=0> sensitivity_sd;   // Between-participant variability sensitivity

    /* participant-level main paameter*/
    real<lower=0, upper=1> alphaAct[nParts];   // Learning rate for Action Learning Value
    real<lower=0, upper=1> alphaClr[nParts];   // Learning rate for Color Learning Value
    real<lower=0, upper=1> weightAct[nParts];  // Wieghtening of Action Learning Value against to Color Learnig Value
    real<lower=0> sensitivity[nParts];         // With a higher sensitivity value θ, choices are more sensitive to value differences

}
transformed parameters {
    real p_push;   // Probability of reward for pushing responce
    real p_pull;   // Probability of reward for pulling responce
    real p_yell;   // Probability of reward for yrllow responce
    real p_blue;   // Probability of reward for blue responce
    real EV_push;  // Standard Expected Value of push action
    real EV_pull;  // Standard Expected Value of pull action
    real EV_yell;  // Standard Expected Value of yellow action
    real EV_blue;  // Standard Expected Value of blue action
    real EV_push_yell;      // Weighting two strategies between push action and yellow color values learning
    real EV_push_blue;      // Weighting two strategies between push action and blue color values learning
    real EV_pull_yell;      // Weighting two strategies between pull action and yellow color values learning
    real EV_pull_blue;      // Weighting two strategies between pull action and blue color values learning
    vector[N] soft_max_EV;  //  The soft-max function for each trial
   
   // Calculating the probability of reward
   for (i in 1:N) {
        if (indicator[i]==1){
            p_push = p_push_init;
            p_pull = 1 - p_push_init;
            p_yell = p_yell_init;
            p_blue = 1 - p_yell_init;
        }
        // Calculating the Standard Expected Value
        EV_push = p_push*winAmtPushable[i];
        EV_pull = p_pull*winAmtPullable[i];
        EV_yell = p_yell*winAmtYellow[i];
        EV_blue = p_blue*winAmtBlue[i];
       
        // Relative contribution of Action Value Learning verus Color Value Learning
        EV_push_yell = weightAct[participant[N]]*EV_push + (1 - weightAct[participant[N]])*EV_yell;
        EV_push_blue = weightAct[participant[N]]*EV_push + (1 - weightAct[participant[N]])*EV_blue;
        EV_pull_yell = weightAct[participant[N]]*EV_pull + (1 - weightAct[participant[N]])*EV_yell;
        EV_pull_blue = weightAct[participant[N]]*EV_pull + (1 - weightAct[participant[N]])*EV_blue;
       
        /* Calculating the soft-max function over weightening Action and Color conditions*/ 
        // pushed and yellow vs pulled and blue
        if ((pushed[i] == 1 && yellowChosen[i] == 1) || (pushed[i] == 0 && yellowChosen[i] == 0))
            soft_max_EV[i] = exp(sensitivity[participant[N]]*EV_push_yell)/(exp(sensitivity[participant[N]]*EV_push_yell) + exp(sensitivity[participant[N]]*EV_pull_blue));

        // pushed and blue vs pulled and yellow
        if ((pushed[i] == 1 && yellowChosen[i] == 0) || (pushed[i] == 0 && yellowChosen[i] == 1))
            soft_max_EV[i] = exp(sensitivity[participant[N]]*EV_push_blue)/(exp(sensitivity[participant[N]]*EV_push_blue) + exp(sensitivity[participant[N]]*EV_pull_yell));  
          
        // RL rule update for computing prediction error and internal value expectation for the next trial based on the current reward output and interal value expectation
        if (pushed[i] == 1){
            p_push = p_push + alphaAct[participant[N]]*(rewarded[i] - p_push); 
            p_pull = 1 - p_push;
        }
        else{
            p_pull = p_pull + alphaAct[participant[N]]*(rewarded[i] - p_pull);
            p_push = 1 - p_pull;
        }    
        if (yellowChosen[i] == 1){
           p_yell = p_yell + alphaClr[participant[N]]*(rewarded[i] - p_yell);
           p_blue = 1 - p_yell;
        }    
        else{
           p_blue = p_blue + alphaClr[participant[N]]*(rewarded[i] - p_blue);
           p_yell = 1 - p_blue;           
        }
    }   
}
model { 
    /* Hierarchical mu parameter*/                               
    sensitivity_hier ~ gamma(1,1); 
    alphaAct_hier ~ normal(0,1) T[0, 1];
    alphaClr_hier ~ normal(0,1) T[0, 1]; 
    weightAct_hier ~ normal(.5,.5) T[0, 1];

    /* Hierarchical sd parameter*/                               
    sensitivity_sd ~ gamma(.1,1); 
    alphaAct_sd ~ gamma(.1,1);  
    alphaClr_sd ~ gamma(.1,1); 
    weightAct_sd ~ gamma(.1,1); 

    /* participant-level main paameter*/
    for (p in 1:nParts) {
        sensitivity[p] ~ normal(sensitivity_hier, sensitivity_sd) T[0, ]; 
        alphaAct[p] ~ normal(alphaAct_hier, alphaAct_sd) T[0, 1]; 
        alphaClr[p] ~ normal(alphaClr_hier, alphaClr_sd) T[0, 1];
        weightAct[p] ~ normal(weightAct_hier, weightAct_sd) T[0, 1];
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