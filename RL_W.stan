/* Model RL in addditon to weightening parameter to modele both Action and Stimulus values learning at the same time
*/ 
data {
    int<lower=1> N;                            // Number of trial-level observations
    int<lower=0, upper=1> pushed[N];           // 1 if pushed and 0 if pulled 
    int<lower=0, upper=1> yellowChosen[N];     // 1 if yellow color is chosen and 0 if yellow color is not chosen 
    int<lower=0, upper=100> winAmtPushable[N]; // The amount of values feedback when pushing is correct response
    int<lower=0, upper=100> winAmtYellow[N];   // The amount of values feedback when yellow chosen is correct response 
    int<lower=0, upper=1> rewarded[N];         // 1 for rewarding and 0 for punishment
    real<lower=0, upper=1> p_push_init;        // Initial value of reward probability for pushed responce
    real<lower=0, upper=1> p_yell_init;        // Initial value of reward probability for stimulus responce
 }
transformed data{
    // Transform the actual choice data to two combined possibilities
    // 1: pushed and yellow vs 0: pulled and blue
    // Or
    // 1: pushed and blue   vs 0: pulled and yellow
    // NaN: no choice/irregular response (e.g. pushing when pulling allowed) have been removed before model fitting
    int<lower=0, upper=1> resActStim[N];
    resActStim = pushed;   
}
parameters {
    real<lower=0, upper=1> alphaAct_; // Learning rate for Action Learning Value
    real<lower=0, upper=1> alphaStim_; // Learning rate for Stimulus Learning Value
    real<lower=0, upper=1> weightAct_;  // Wieghtening of Action Learning Value against to Color Learnig Value
    real<lower=0> beta_; // With a higher sensitivity value θ, choices are more sensitive to value differences 
 
}
transformed parameters {
   real<lower=0, upper=1> p_push;  // Probability of reward for pushing responce
   real<lower=0, upper=1> p_pull;  // Probability of reward for pulling responce
   real<lower=0, upper=1> p_yell;  // Probability of reward for yrllow responce
   real<lower=0, upper=1> p_blue;  // Probability of reward for blue responce
   real EV_push;  // Standard Expected Value of push action
   real EV_pull;  // Standard Expected Value of pull action
   real EV_yell;  // Standard Expected Value of yellow action
   real EV_blue;  // Standard Expected Value of blue action
   real EV_push_yell;  // Weighting two strategies between push action and yellow color values learning
   real EV_push_blue;  // Weighting two strategies between push action and blue color values learning
   real EV_pull_yell;  // Weighting two strategies between pull action and yellow color values learning
   real EV_pull_blue;  // Weighting two strategies between pull action and blue color values learning
   vector[N] soft_max_EV; //  The soft-max function for each trial
   
   // Calculating the probability of reward
   p_push = p_push_init;
   p_pull = 1 - p_push_init;
   p_yell = p_yell_init;
   p_blue = 1 - p_yell_init;
   for (i in 1:N) {
       // RL rule update
       if (pushed[i] == 1){
            p_push = p_push + alphaAct_*(rewarded[i] - p_push); 
            p_pull = 1 - p_push;
        }
       else{
            p_pull = p_pull + alphaAct_*(rewarded[i] - p_pull);
            p_push = 1 - p_pull;
       }    
       if (yellowChosen[i] == 1){
           p_yell = p_yell + alphaStim_*(rewarded[i] - p_yell);
           p_blue = 1 - p_yell;
       }    
       else{
           p_blue = p_blue + alphaStim_*(rewarded[i] - p_blue);
           p_yell = 1 - p_blue;           
       }
       // Calculating the Standard Expected Value
       EV_push = p_push*winAmtPushable[i];
       EV_pull = p_pull*(100 - winAmtPushable[i]);
       EV_yell = p_yell*winAmtYellow[i];
       EV_blue = p_blue*(100 - winAmtYellow[i]);
       
       // Relative contribution of Action Value Learning verus Stimulus Value Learning
       EV_push_yell = weightAct_*EV_push + (1 - weightAct_)*EV_yell;
       EV_push_blue = weightAct_*EV_push + (1 - weightAct_)*EV_blue;
       EV_pull_yell = weightAct_*EV_pull + (1 - weightAct_)*EV_yell;
       EV_pull_blue = weightAct_*EV_pull + (1 - weightAct_)*EV_blue;
       
        /* Calculating the soft-max function ovwer weightening Action and Color conditions*/ 
        // pushed and yellow vs pulled and blue
        if ((pushed[i] == 1 && yellowChosen[i] == 1) || (pushed[i] == 0 && yellowChosen[i] == 0))
            soft_max_EV[i] = exp(beta_*EV_push_yell)/(exp(beta_*EV_push_yell) + exp(beta_*EV_pull_blue));

        // pushed and blue vs pulled and yellow
        if ((pushed[i] == 1 && yellowChosen[i] == 0) || (pushed[i] == 0 && yellowChosen[i] == 1))
            soft_max_EV[i] = exp(beta_*EV_push_blue)/(exp(beta_*EV_push_blue) + exp(beta_*EV_pull_yell));      
    }   
}
model {
    /* learning rate parameters prior */
    alphaAct_ ~ beta(3,3); 
    alphaStim_ ~ beta(3,3); 

    /* Wieghtening parameter prior */
    weightAct_ ~ beta(3,3); 
    
    /* sensitivity parameter prior */
    beta_ ~ gamma(1,1) T[0, 10];     
    
    /* RL likelihood */
    for (i in 1:N) { 
        resActStim[i] ~ bernoulli(soft_max_EV[i]);
    }
}
generated quantities { 
   vector[N] log_lik;  
   
    /*  RL Log density likelihood */
    for (i in 1:N) {
         log_lik[i] = bernoulli_lpmf(resActStim[i] | soft_max_EV[i]);
   }
}