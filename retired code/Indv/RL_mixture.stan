/* Mixure RL*/ 
data {
    int<lower=1> N;        // Number of trial-level observations
    array[N] int<lower=0, upper=1> pushed;             // 1 if pushed and 0 if pulled 
    array[N] int<lower=0, upper=1> yellowChosen;       // 1 if yellow color is chosen and 0 if yellow color is not chosen 
    array[N] real<lower=0, upper=100> winAmtPushable;  // The amount of values feedback when pushing is correct response
    array[N] real<lower=0, upper=100> winAmtPullable;  // The amount of values feedback when pulling is correct response
    array[N] real<lower=0, upper=100> winAmtYellow;    // The amount of values feedback when yellow chosen is correct response 
    array[N] real<lower=0, upper=100> winAmtBlue;      // The amount of values feedback when blue chosen is correct response 
    array[N] int<lower=0, upper=1> rewarded;           // 1 for rewarding and 0 for punishment
    array[N] int<lower=1> indicator;                   // indicator of the first trial of each participant, the first is denoted 1 otherwise 0
    real<lower=0, upper=1> p_push_init;         // Initial value of reward probability for pushed responce
    real<lower=0, upper=1> p_yell_init;         // Initial value of reward probability for Color responce
 }
parameters {
    real alphaAct;     // Learning rate for Action Learning Value
    real alphaClr;     // Learning rate for Color Learning Value
    real sensitivity;  // With a higher sensitivity value θ, choices are more sensitive to value differences
    simplex[2] theta;  // mixing proportions
}
transformed parameters {
    real p_push;  // Probability of reward for pushing responce
    real p_pull;  // Probability of reward for pulling responce
    real p_yell;  // Probability of reward for yrllow responce
    real p_blue;  // Probability of reward for blue responce
    real EV_push;  // Standard Expected Value of push action
    real EV_pull;  // Standard Expected Value of pull action
    real EV_yell;  // Standard Expected Value of yellow action
    real EV_blue;  // Standard Expected Value of blue action
    vector[N] soft_max_act;  //  The soft-max function for each trial for Acion values
    vector[N] soft_max_clr;  //  The soft-max function for each trial for color values
   
    // Calculating the probability of reward
    for (i in 1:N) {
        if (indicator[i]==1){
            p_push = p_push_init;
            p_pull = 1 - p_push_init;
            p_yell = p_yell_init;
            p_blue = 1 - p_yell_init;
        }
        
        // Action-based decision  
        // Calculating the Standard Expected Value
        EV_push = p_push*winAmtPushable[i];
        EV_pull = p_pull*winAmtPullable[i];
        // soft max function
        soft_max_act[i] = exp(sensitivity*EV_push)/(exp(sensitivity*EV_push) + exp(sensitivity*EV_pull));

        // Color-based decision  
        // Calculating the Standard Expected Value
        EV_yell = p_yell*winAmtYellow[i];
        EV_blue = p_blue*winAmtBlue[i];
        // soft max function 
        soft_max_clr[i] = exp(sensitivity*EV_yell)/(exp(sensitivity*EV_yell) + exp(sensitivity*EV_blue));  
   
        // RL rule update for computing prediction error and internal value expectation for the next trial based on the current reward output and interal value expectation
        if (pushed[i] == 1){
            p_push = p_push + alphaAct*(rewarded[i] - p_push); 
            p_pull = 1 - p_push;
        }
        else{
            p_pull = p_pull + alphaAct*(rewarded[i] - p_pull);
            p_push = 1 - p_pull;
        }    
        if (yellowChosen[i] == 1){
           p_yell = p_yell + alphaClr*(rewarded[i] - p_yell);
           p_blue = 1 - p_yell;
        }    
        else{
           p_blue = p_blue + alphaClr*(rewarded[i] - p_blue);
           p_yell = 1 - p_blue;           
        } 
    }   
}
model {
    /* prior */
    sensitivity ~ gamma(1,1); 
    alphaAct ~ beta(1, 1); 
    alphaClr ~ beta(1, 1);
    
    vector[2] log_theta = log(theta);  // cache log calculation
    /* RL likelihood */
    for (i in 1:N) {         
        // Relative contribution of Action Value Learning verus Color Value Learning
        vector[2] lps = log_theta;
        lps[1] += binomial_lpmf(pushed[i] | 1, soft_max_act[i]);
        lps[2] += binomial_lpmf(yellowChosen[i] | 1, soft_max_clr[i]);
        // target
        target += log_sum_exp(lps);
    }
}
generated quantities { 
    // save log likelihood
    vector[N] log_lik;
      
    vector[2] log_theta = log(theta);  // cache log calculation
    /*  RL Log density likelihood */
    for (i in 1:N) { 
        // Relative contribution of Action Value Learning verus Color Value Learning
        vector[2] lps = log_theta;
        lps[1] += binomial_lpmf(pushed[i] | 1, soft_max_act[i]);
        lps[2] += binomial_lpmf(yellowChosen[i] | 1, soft_max_clr[i]);
        // target
        log_lik[i] = log_sum_exp(lps);
    }
}