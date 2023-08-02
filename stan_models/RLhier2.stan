/* Model RL in addditon to weightening parameter to modele both Action and Color values learning at the same time
   This code can only model each singel data for instance one condition in one specific session and run.
   Therefore, this code conducts the individual level nanalysis
*/ 
data {
    int<lower=1> N;       // Number of trial-level observations
    int<lower=1> nParts;  // Number of participants
    int<lower=1> nCond;   // Number of conditions (Action vs Color)
    int<lower=1> nSes;    // Number of Session
    int<lower=0, upper=1> pushed[N];           // 1 if pushed and 0 if pulled 
    int<lower=0, upper=1> yellowChosen[N];     // 1 if yellow color is chosen and 0 if yellow color is not chosen 
    int<lower=0, upper=100> winAmtPushable[N]; // The amount of values feedback when pushing is correct response
    int<lower=0, upper=100> winAmtYellow[N];   // The amount of values feedback when yellow chosen is correct response 
    int<lower=0, upper=1> rewarded[N];         // 1 for rewarding and 0 for punishment
    real<lower=0, upper=1> p_push_init;        // Initial value of reward probability for pushed responce
    real<lower=0, upper=1> p_yell_init;        // Initial value of reward probability for Color responce
    int<lower=1> participant[N];   // participant index for each trial
    int<lower=1, upper=2> session[N];          // session index for each trial
    int<lower=1, upper=2> condition[N];             // Condition index for each trial, 1: Action, 2: Stimulus
 }
parameters {
 
    /* Hierarchical mu parameter*/                               
    real alphaAct_aHier[nSes, nCond];     // Hierarchical Learning rate in  Action Learning Value
    real alphaClr_aHier[nSes, nCond];     // Hierarchical Learning rate in  Color Learning Value  
    real weightAct_aHier[nSes, nCond];    // Hierarchical Wieghtening in  Action Learning Value against to Color Learnig Value
    real sensitivity_aHier[nSes];          // Hierarchical variability Sensitivity
    
    /* Hierarchical mu parameter*/                               
    real alphaAct_bHier[nSes, nCond];      // Hierarchical Learning rate in  Action Learning Value
    real alphaClr_bHier[nSes, nCond];      // Hierarchical Learning rate in  Color Learning Value  
    real weightAct_bHier[nSes, nCond];     // Hierarchical Wieghtening in  Action Learning Value against to Color Learnig Value
    real<lower=0, upper=1> sensitivity_bHier[nSes];          // Hierarchical variability Sensitivity

    /* participant-level main paameter*/
    real alphaAct[nParts, nSes, nCond];   // Individual Learning rate in  Action Learning Value
    real alphaClr[nParts, nSes, nCond];   // Individual Learning rate in Color Learning Value
    real weightAct[nParts, nSes, nCond];  // Individual Wieghtening in Action Learning Value against to Color Learnig Value
    real<lower=0, upper=1> sensitivity[nParts, nSes];  // Individual Sensitivity,  With a higher sensitivity value θ, choices are more sensitive to value differences 
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
            p_push = p_push + alphaAct[participant[i], session[i], condition[i]]*(rewarded[i] - p_push); 
            p_pull = 1 - p_push;
        }
       else{
            p_pull = p_pull + alphaAct[participant[i], session[i], condition[i]]*(rewarded[i] - p_pull);
            p_push = 1 - p_pull;
       }    
       if (yellowChosen[i] == 1){
           p_yell = p_yell + alphaClr[participant[i], session[i], condition[i]]*(rewarded[i] - p_yell);
           p_blue = 1 - p_yell;
       }    
       else{
           p_blue = p_blue + alphaClr[participant[i], session[i], condition[i]]*(rewarded[i] - p_blue);
           p_yell = 1 - p_blue;           
       }
       // Calculating the Standard Expected Value
       EV_push = p_push*winAmtPushable[i];
       EV_pull = p_pull*(100 - winAmtPushable[i]);
       EV_yell = p_yell*winAmtYellow[i];
       EV_blue = p_blue*(100 - winAmtYellow[i]);
       
       // Relative contribution of Action Value Learning verus Color Value Learning
       EV_push_yell = weightAct[participant[i], session[i], condition[i]]*EV_push + (1 - weightAct[participant[i], session[i], condition[i]])*EV_yell;
       EV_push_blue = weightAct[participant[i], session[i], condition[i]]*EV_push + (1 - weightAct[participant[i], session[i], condition[i]])*EV_blue;
       EV_pull_yell = weightAct[participant[i], session[i], condition[i]]*EV_pull + (1 - weightAct[participant[i], session[i], condition[i]])*EV_yell;
       EV_pull_blue = weightAct[participant[i], session[i], condition[i]]*EV_pull + (1 - weightAct[participant[i], session[i], condition[i]])*EV_blue;
       
        /* Calculating the soft-max function ovwer weightening Action and Color conditions*/ 
        // pushed and yellow vs pulled and blue
        if ((pushed[i] == 1 && yellowChosen[i] == 1) || (pushed[i] == 0 && yellowChosen[i] == 0))
            soft_max_EV[i] = exp(sensitivity[participant[i], session[i]]*EV_push_yell)/(exp(sensitivity[participant[i], session[i]]*EV_push_yell) + exp(sensitivity[participant[i], session[i]]*EV_pull_blue));

        // pushed and blue vs pulled and yellow
        if ((pushed[i] == 1 && yellowChosen[i] == 0) || (pushed[i] == 0 && yellowChosen[i] == 1))
            soft_max_EV[i] = exp(sensitivity[participant[i], session[i]]*EV_push_blue)/(exp(sensitivity[participant[i], session[i]]*EV_push_blue) + exp(sensitivity[participant[i], session[i]]*EV_pull_yell));      
    }   
}
model {
   
     /* Hierarchical mu parameter*/                               
    for (s in 1:nSes) {
      for (c in 1:nCond) {
        alphaAct_aHier[s, c] ~ gamma(1, 1); 
        alphaClr_aHier[s, c] ~ gamma(1, 1);
        weightAct_aHier[s, c] ~ gamma(1, 1); 
        
        
        alphaAct_bHier[s, c] ~ gamma(1, 1); 
        alphaClr_bHier[s, c] ~ gamma(1, 1);
        weightAct_bHier[s, c] ~ gamma(1, 1); 
        
       }
      sensitivity_aHier[s] ~ gamma(1, 1);   
      sensitivity_bHier[s] ~ gamma(1, 1);   
    } 
       
    /* participant-level main paameter*/
    for (p in 1:nParts) {
      for (s in 1:nSes) {
        for (c in 1:nCond) {
          alphaAct[p, s, c] ~ beta(alphaAct_aHier[s, c], alphaAct_bHier[s, c]); 
          alphaClr[p, s, c] ~ beta(alphaClr_aHier[s, c], alphaClr_bHier[s, c]); 
          weightAct[p, s, c] ~ beta(weightAct_aHier[s, c], weightAct_bHier[s, c]); 
       }
       sensitivity[p, s] ~ normal(sensitivity_aHier[s], sensitivity_bHier[s]) T[0,10];   
      }
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