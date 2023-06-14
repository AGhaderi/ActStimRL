/* Model RL in addditon to weightening parameter to modele both Action and Color values learning at the same time
   This code can only model each singel data for instance one condition in one specific session and run.
   Therefore, this code conducts the individual level nanalysis
*/ 
data {
    int<lower=1> N;                            // Number of trial-level observations
    int<lower=1> nCond;   // Number of conditions (Action vs Color)
    int<lower=1> nSes;    // Number of Session
    int<lower=0, upper=1> pushed[N];           // 1 if pushed and 0 if pulled 
    int<lower=0, upper=1> yellowChosen[N];     // 1 if yellow color is chosen and 0 if yellow color is not chosen 
    int<lower=0, upper=100> winAmtPushable[N]; // The amount of values feedback when pushing is correct response
    int<lower=0, upper=100> winAmtYellow[N];   // The amount of values feedback when yellow chosen is correct response 
    int<lower=0, upper=1> rewarded[N];         // 1 for rewarding and 0 for punishment
    real<lower=0, upper=1> p_push_init;        // Initial value of reward probability for pushed responce
    real<lower=0, upper=1> p_yell_init;        // Initial value of reward probability for Color responce
    real<lower=1, upper=2> session[N];         // Indices of session for each trials
    real<lower=1, upper=2> cond[N];            // Indices of cindition for each trials, 1: Action, 2: Stimulus
 }
transformed data{
    // Transform the actual choice data to two combined possibilities
    // 1: pushed and yellow vs 0: pulled and blue
    // Or
    // 1: pushed and blue   vs 0: pulled and yellow
    // NaN: no choice/irregular response (e.g. pushing when pulling allowed) have been removed before model fitting
    int<lower=0, upper=1> resActClr[N];
    resActClr = pushed;
}
parameters {
    matrix<lower=0, upper=1>[nSes, nCond] alphaAct_; // Learning rate for Action Learning Value
    matrix<lower=0, upper=1>[nSes, nCond] alphaClr_; // Learning rate for Color Learning Value
    matrix<lower=0, upper=1>[nSes, nCond] weightAct_;  // Wieghtening of Action Learning Value against to Color Learnig Value
    vector<lower=0>[nSes] beta_; // With a higher sensitivity value θ, choices are more sensitive to value differences 
 
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
            p_push = p_push + alphaAct_[session[i], cond[i]]*(rewarded[i] - p_push); 
            p_pull = 1 - p_push;
        }
       else{
            p_pull = p_pull + alphaAct_[session[i], cond[i]]*(rewarded[i] - p_pull);
            p_push = 1 - p_pull;
       }    
       if (yellowChosen[i] == 1){
           p_yell = p_yell + alphaClr_[session[i], cond[i]]*(rewarded[i] - p_yell);
           p_blue = 1 - p_yell;
       }    
       else{
           p_blue = p_blue + alphaClr_[session[i], cond[i]]*(rewarded[i] - p_blue);
           p_yell = 1 - p_blue;           
       }
       // Calculating the Standard Expected Value
       EV_push = p_push*winAmtPushable[i];
       EV_pull = p_pull*(100 - winAmtPushable[i]);
       EV_yell = p_yell*winAmtYellow[i];
       EV_blue = p_blue*(100 - winAmtYellow[i]);
       
       // Relative contribution of Action Value Learning verus Color Value Learning
       EV_push_yell = weightAct_[session[i], cond[i]]*EV_push + (1 - weightAct_[session[i], cond[i]])*EV_yell;
       EV_push_blue = weightAct_[session[i], cond[i]]*EV_push + (1 - weightAct_[session[i], cond[i]])*EV_blue;
       EV_pull_yell = weightAct_[session[i], cond[i]]*EV_pull + (1 - weightAct_[session[i], cond[i]])*EV_yell;
       EV_pull_blue = weightAct_[session[i], cond[i]]*EV_pull + (1 - weightAct_[session[i], cond[i]])*EV_blue;
       
        /* Calculating the soft-max function ovwer weightening Action and Color conditions*/ 
        // pushed and yellow vs pulled and blue
        if ((pushed[i] == 1 && yellowChosen[i] == 1) || (pushed[i] == 0 && yellowChosen[i] == 0))
            soft_max_EV[i] = exp(beta_[session[i]]*EV_push_yell)/(exp(beta_[session[i]]*EV_push_yell) + exp(beta_[session[i]]*EV_pull_blue));

        // pushed and blue vs pulled and yellow
        if ((pushed[i] == 1 && yellowChosen[i] == 0) || (pushed[i] == 0 && yellowChosen[i] == 1))
            soft_max_EV[i] = exp(beta_[session[i]]*EV_push_blue)/(exp(beta_[session[i]]*EV_push_blue) + exp(beta_[session[i]]*EV_pull_yell));      
    }   
}
model {
     /* sensitivity parameter prior */
     /* learning rate parameters prior */
    /* Wieghtening parameter prior */   
    for (s in 1:nSes) {
      
      beta_[s] ~ gamma(1,1) T[0, 10];   
      
      for (r in 1:nCond) {
        alphaAct_[s, r] ~ beta(3,3); 
        alphaClr_[s, r] ~ beta(3,3);
        weightAct_ ~ beta(3,3); 
       }
    }
    
    /* RL likelihood */
    for (i in 1:N) { 
        resActClr[i] ~ bernoulli(soft_max_EV[i]);
    }
}
generated quantities { 
   vector[N] log_lik;  
   
    /*  RL Log density likelihood */
    for (i in 1:N) {
         log_lik[i] = bernoulli_lpmf(resActClr[i] | soft_max_EV[i]);
   }
}