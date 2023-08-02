/* Hierarchical Reinforcment leanring based on Rescolda and Wagner rule learning plus new weightening parameter to modele both Action and Color values learning at the same time
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
    int<lower=1> participant[N];               // participant index for each trial
    int<lower=1, upper=2> session[N];          // session index for each trial
    int<lower=1, upper=2> condition[N];        // Condition index for each trial, 1: Action, 2: Stimulus
 }
parameters {
      /* sigma paameter*/
    real<lower=0> alphaAct_sd;      // Between-participant variability Learning rate in  Action Learning Value
    real<lower=0> alphaClr_sd;      // Between-participant variability Learning rate in  Color Learning Value  
    real<lower=0> weightAct_sd;     // Between-participant variability Wieghtening in  Action Learning Value against to Color Learnig Value
    real<lower=0> sensitivity_sd;   // Between-participant variability Sensitivity

    /* Hierarchical mu parameter*/                               
    real alphaAct_hier[nSes, nCond];   // Hierarchical Learning rate in  Action Learning Value
    real alphaClr_hier[nSes, nCond];   // Hierarchical Learning rate in  Color Learning Value  
    real weightAct_hier[nSes, nCond];  // Hierarchical Wieghtening in  Action Learning Value against to Color Learnig Value
    real sensitivity_hier[nSes];       // Hierarchical variability Sensitivity

    /* participant-level main paameter*/
    real alphaAct[nParts, nSes, nCond];   // Individual Learning rate in  Action Learning Value
    real alphaClr[nParts, nSes, nCond];   // Individual Learning rate in Color Learning Value
    real weightAct[nParts, nSes, nCond];  // Individual Wieghtening in Action Learning Value against to Color Learnig Value
    real sensitivity[nParts, nSes];       // Individual Sensitivity,  With a higher sensitivity value θ, choices are more sensitive to value differences 
}
transformed parameters {
    // Trnasformation of hierarchical parameters
    real transf_alphaAct_hier[nSes, nCond];   
    real transf_alphaClr_hier[nSes, nCond];       
    real transf_weightAct_hier[nSes, nCond];     
    real transf_sensitivity_hier[nSes];
    
    // Trnasformation of individual parameters
    real transf_alphaAct[nParts, nSes, nCond]; 
    real transf_alphaClr[nParts, nSes, nCond];  
    real transf_weightAct[nParts, nSes, nCond];   
    real transf_sensitivity[nParts, nSes];  
  
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

    // Transformation of hierarchcial parameters
    for (s in 1:nSes) {
      for (c in 1:nCond) { 	  
        transf_alphaAct_hier[s, c] = Phi(alphaAct_hier[s, c]);   
        transf_alphaClr_hier[s, c] = Phi(alphaClr_hier[s, c]);      
        transf_weightAct_hier[s, c] = Phi(weightAct_hier[s, c]);    
      }  
      transf_sensitivity_hier[s] = log(1 + exp(sensitivity_hier[s]));
    }
    
   // Transformation of idividual parameters
 	  for (p in 1:nParts) {
      for (s in 1:nSes) {
        for (c in 1:nCond) {
          transf_alphaAct[p, s, c] = Phi(alphaAct[p, s, c]);   
          transf_alphaClr[p, s, c] = Phi(alphaClr[p, s, c]);      
          transf_weightAct[p, s, c] = Phi(weightAct[p, s, c]);
        }
        transf_sensitivity[p, s] = log(1+exp(sensitivity[p, s]));   
      }
	  }
	  
   // Calculating the intitual the probability of reward
   p_push = p_push_init;
   p_pull = 1 - p_push_init;
   p_yell = p_yell_init;
   p_blue = 1 - p_yell_init;
   // loop over trials
   for (i in 1:N) {
       // Calculating the Standard Expected Value
       EV_push = p_push*winAmtPushable[i];
       EV_pull = p_pull*(100 - winAmtPushable[i]);
       EV_yell = p_yell*winAmtYellow[i];
       EV_blue = p_blue*(100 - winAmtYellow[i]);
       
       // Relative contribution of Action Value Learning verus Color Value Learning as a linear combination
       EV_push_yell = transf_weightAct[participant[i], session[i], condition[i]]*EV_push + (1 - transf_weightAct[participant[i], session[i], condition[i]])*EV_yell;
       EV_push_blue = transf_weightAct[participant[i], session[i], condition[i]]*EV_push + (1 - transf_weightAct[participant[i], session[i], condition[i]])*EV_blue;
       EV_pull_yell = transf_weightAct[participant[i], session[i], condition[i]]*EV_pull + (1 - transf_weightAct[participant[i], session[i], condition[i]])*EV_yell;
       EV_pull_blue = transf_weightAct[participant[i], session[i], condition[i]]*EV_pull + (1 - transf_weightAct[participant[i], session[i], condition[i]])*EV_blue;
      
        /* Calculating the soft-max function over weightening Action and Color conditions*/ 
        // pushed and yellow vs pulled and blue
        if ((pushed[i] == 1 && yellowChosen[i] == 1) || (pushed[i] == 0 && yellowChosen[i] == 0))
            soft_max_EV[i] = exp(transf_sensitivity[participant[i], session[i]]*EV_push_yell)/(exp(transf_sensitivity[participant[i], session[i]]*EV_push_yell) + exp(transf_sensitivity[participant[i], session[i]]*EV_pull_blue));

        // pushed and blue vs pulled and yellow
        else if ((pushed[i] == 1 && yellowChosen[i] == 0) || (pushed[i] == 0 && yellowChosen[i] == 1))
            soft_max_EV[i] = exp(transf_sensitivity[participant[i], session[i]]*EV_push_blue)/(exp(transf_sensitivity[participant[i], session[i]]*EV_push_blue) + exp(transf_sensitivity[participant[i], session[i]]*EV_pull_yell));      

       // RL rule update
       if (pushed[i] == 1){
            p_push = p_push + transf_alphaAct[participant[i], session[i], condition[i]]*(rewarded[i] - p_push); 
            p_pull = 1 - p_push;
        }
       else{
            p_pull = p_pull + transf_alphaAct[participant[i], session[i], condition[i]]*(rewarded[i] - p_pull);
            p_push = 1 - p_pull;
       }    
       if (yellowChosen[i] == 1){
           p_yell = p_yell + transf_alphaClr[participant[i], session[i], condition[i]]*(rewarded[i] - p_yell);
           p_blue = 1 - p_yell;
       }    
       else{
           p_blue = p_blue + transf_alphaClr[participant[i], session[i], condition[i]]*(rewarded[i] - p_blue);
           p_yell = 1 - p_blue;           
       }
    }   
}
model {
  
    /* sigma paameter*/
    alphaAct_sd ~ gamma(.1, .1); 
    alphaClr_sd ~ gamma(.1, .1); 
    weightAct_sd ~ gamma(.1, .1); 
    sensitivity_sd ~ gamma(.1, .1);     
   
     /* Hierarchical mu parameter*/                               
    for (s in 1:nSes) {
      for (c in 1:nCond) {
        alphaAct_hier[s, c] ~ normal(0, 1); 
        alphaClr_hier[s, c] ~ normal(0, 1);
        weightAct_hier[s, c] ~ normal(0, 1); 
       }
      sensitivity_hier[s] ~ normal(0, 1);   
    } 
       
    /* participant-level main paameter*/
    for (p in 1:nParts) {
      for (s in 1:nSes) {
        for (c in 1:nCond) {
          alphaAct[p, s, c] ~ normal(transf_alphaAct_hier[s, c], alphaAct_sd); 
          alphaClr[p, s, c] ~ normal(transf_alphaClr_hier[s, c], alphaClr_sd); 
          weightAct[p, s, c] ~ normal(transf_weightAct_hier[s, c], weightAct_sd); 
       }
       sensitivity[p, s] ~ normal(transf_sensitivity_hier[s], sensitivity_sd);   
      }
    }
    
    /* RL likelihood */
    for (i in 1:N) { 
        pushed ~ bernoulli(soft_max_EV[i]);
        //print("target = ", target(), " when soft_max_EV = ", soft_max_EV[i], ", transf_sensitivity = ", transf_sensitivity);
    }
}
generated quantities { 
   vector[N] log_lik;  
   
    /*  RL Log density likelihood */
    for (i in 1:N) {
         log_lik[i] = bernoulli_lpmf(pushed[i] | soft_max_EV[i]);
   }
}