/* Model RL in addditon to weightening parameter to modele both Action and Color values learning at the same time
   This code can only model each singel data for instance one condition in one specific session and run.
   Therefore, this code conducts the individual level nanalysis
*/ 
data {
    int<lower=1> N;       // Number of trial-level observations
    int<lower=1> nparts;  // Number of participants
    int<lower=1> nCond;   // Number of conditions (Action vs Color)
    int<lower=1> nSes;    // Number of Session
    int<lower=0, upper=1> pushed[N];           // 1 if pushed and 0 if pulled 
    int<lower=0, upper=1> yellowChosen[N];     // 1 if yellow color is chosen and 0 if yellow color is not chosen 
    int<lower=0, upper=100> winAmtPushable[N]; // The amount of values feedback when pushing is correct response
    int<lower=0, upper=100> winAmtYellow[N];   // The amount of values feedback when yellow chosen is correct response 
    int<lower=0, upper=1> rewarded[N];         // 1 for rewarding and 0 for punishment
    real<lower=0, upper=1> p_push_init;        // Initial value of reward probability for pushed responce
    real<lower=0, upper=1> p_yell_init;        // Initial value of reward probability for Color responce
    int<lower=1> participant[N_obs + N_mis];   // participant index for each trial
    int<lower=1, upper=2> session[N];          // session index for each trial
    int<lower=1, upper=2> cond[N];             // Condition index for each trial, 1: Action, 2: Stimulus
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
      /* sigma paameter*/
    real<lower=0> alphaAct_sd;      // Between-participant variability Learning rate in  Action Learning Value
    real<lower=0> alphaClr_sd;      // Between-participant variability Learning rate in  Color Learning Value  
    real<lower=0> weightAct_sd;     // Between-participant variability Wieghtening in  Action Learning Value against to Color Learnig Value
    real<lower=0> beta_sd;          // Between-participant variability Sensitivity

    /* Hierarchical mu parameter*/                               
    real<lower=0, upper=1> alphaAct_hier;      // Hierarchical Learning rate in  Action Learning Value
    real<lower=0, upper=1> alphaClr_hier;      // Hierarchical Learning rate in  Color Learning Value  
    real<lower=0, upper=1> weightAct_hier;     // Hierarchical Wieghtening in  Action Learning Value against to Color Learnig Value
    real<lower=0, upper=1> beta_hier;          // Hierarchical variability Sensitivity


    /* participant-level main paameter*/
    matrix<lower=0, upper=1>[nparts, nSes, nCond] alphaAct_;   // Individual Learning rate in  Action Learning Value
    matrix<lower=0, upper=1>[nparts, nSes, nCond] alphaClr_;   // Individual Learning rate in Color Learning Value
    matrix<lower=0, upper=1>[nparts, nSes, nCond] weightAct_;  // Individual Wieghtening in Action Learning Value against to Color Learnig Value
    vector<lower=0, upper=1>[nparts, nSes] beta_;  // Individual Sensitivity,  With a higher sensitivity value θ, choices are more sensitive to value differences 
 
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
            p_push = p_push + alphaAct_[participant[i], session[i], cond[i]]*(rewarded[i] - p_push); 
            p_pull = 1 - p_push;
        }
       else{
            p_pull = p_pull + alphaAct_[participant[i], session[i], cond[i]]*(rewarded[i] - p_pull);
            p_push = 1 - p_pull;
       }    
       if (yellowChosen[i] == 1){
           p_yell = p_yell + alphaClr_[participant[i], session[i], cond[i]]*(rewarded[i] - p_yell);
           p_blue = 1 - p_yell;
       }    
       else{
           p_blue = p_blue + alphaClr_[participant[i], session[i], cond[i]]*(rewarded[i] - p_blue);
           p_yell = 1 - p_blue;           
       }
       // Calculating the Standard Expected Value
       EV_push = p_push*winAmtPushable[i];
       EV_pull = p_pull*(100 - winAmtPushable[i]);
       EV_yell = p_yell*winAmtYellow[i];
       EV_blue = p_blue*(100 - winAmtYellow[i]);
       
       // Relative contribution of Action Value Learning verus Color Value Learning
       EV_push_yell = weightAct_[participant[i], session[i], cond[i]]*EV_push + (1 - weightAct_[participant[i], session[i], cond[i]])*EV_yell;
       EV_push_blue = weightAct_[participant[i], session[i], cond[i]]*EV_push + (1 - weightAct_[participant[i], session[i], cond[i]])*EV_blue;
       EV_pull_yell = weightAct_[participant[i], session[i], cond[i]]*EV_pull + (1 - weightAct_[participant[i], session[i], cond[i]])*EV_yell;
       EV_pull_blue = weightAct_[participant[i], session[i], cond[i]]*EV_pull + (1 - weightAct_[participant[i], session[i], cond[i]])*EV_blue;
       
        /* Calculating the soft-max function ovwer weightening Action and Color conditions*/ 
        // pushed and yellow vs pulled and blue
        if ((pushed[i] == 1 && yellowChosen[i] == 1) || (pushed[i] == 0 && yellowChosen[i] == 0))
            soft_max_EV[i] = exp(beta_[participant[i], session[i]]*EV_push_yell)/(exp(beta_[participant[i], session[i]]*EV_push_yell) + exp(beta_[participant[i], session[i]]*EV_pull_blue));

        // pushed and blue vs pulled and yellow
        if ((pushed[i] == 1 && yellowChosen[i] == 0) || (pushed[i] == 0 && yellowChosen[i] == 1))
            soft_max_EV[i] = exp(beta_[participant[i], session[i]]*EV_push_blue)/(exp(beta_[participant[i], session[i]]*EV_push_blue) + exp(beta_[participant[i], session[i]]*EV_pull_yell));      
    }   
}
model {
  
  
    /* sigma paameter*/
    alphaAct_sd ~ gamma(1,1); 
    alphaClr_sd ~ gamma(1,1); 
    weightAct_sd ~ gamma(1,1); 
    beta_sd ~ gamma(.1,1);    

   
     /* Hierarchical mu parameter*/                               
    for (s in 1:nSes) {
      for (r in 1:nCond) {
        alphaAct_hier[s, r] ~ beta(3,3); 
        alphaClr_hier[s, r] ~ beta(3,3);
        weightAct_hier[s, r] ~ beta(3,3); 
       }
      beta_hier[s] ~ gamma(1,1) T[0, 10];   
    }
    
    
       
    /* participant-level main paameter*/
    for (p in 1:nparts) {
      for (s in 1:nSes) {
        for (r in 1:nCond) {
          alphaAct_[p, s, r] ~ normal(alphaAct_hier[s, r], alphaAct_sd) T[0, 1]; 
          alphaClr_[p, s, r] ~ normal(alphaClr_hier[s, r], alphaClr_sd) T[0, 1]; 
          weightAct_[p, s, r] ~ normal(weightAct_hier[s, r], weightAct_sd) T[0, 1]; 
       }
       beta_hier[p, s] ~ normal(alphaAct_hier[s, r], beta_sd) T[0, 10];   
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