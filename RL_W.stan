/* Model RL in addditon to weightening for modeling both Action and Color values learning at the same time
* This model is just based on responce choices rather that both responce choices and responce time
*/ 
data {
    int<lower=1> N;                          // Number of trial-level observations
    int<lower=0, upper=1> pushedChosen;      // 1 if push acion is chosen and 0 if push action is not chosen 
    int<lower=0, upper=1> yellowChosen;      // 1 if yellow color is chosen and 0 if yellow color is not chosen 
    int<lower=0, upper=100> winAmtPushable;  // the amount of values feedback when push action is correct response
    int<lower=0, upper=100> winAmtYellow;    // 1 if yellow color is chosen and 0 if yellow color is not chosen 
    real<lower=0, upper=1> acc[N_obs];         // accuracy (0, 1) incorrect and correct responses respectively
}
parameters {
    real<lower=0, upper=1> alpha_A; // Learning rate for Action Learning Value
    real<lower=0, upper=1> alpha_C; // Learning rate for Color Learning Value
    real<lower=0, upper=1> weight;  // Wieghtening of Action Learning Value against to Color Learnig Value
    real<lower=0> bet;              // With a higher sensitivity value θ, choices are more sensitive to value differences 
 
}
transformed parameters {
   real<lower=0, upper=1> p_push = .5,;  // Probability of reward for push action
   real<lower=0, upper=1> p_yell = .5,;  // Probability of reward for yellow color
   real EV_push;  // Standard Expected Value of push action
   real EV_pull;  // Standard Expected Value of pull action
   real EV_yell;  // Standard Expected Value of yellow action
   real EV_blue;  // Standard Expected Value of blue action
   real EV_push_yell;  // Weighting two strategies between push action and yellow color values learning
   real EV_push_blue;  // Weighting two strategies between push action and blue color values learning
   real EV_pull_yell;  // Weighting two strategies between pull action and yellow color values learning
   real EV_pull_blue;  // Weighting two strategies between pull action and blue color values learning
   vector[N] soft_max_EV; //  The soft-max function for each trial
   
   # Calculating the probability of reward
   for (i in 1:N) {
       # RL rule update
       p_push = p_push + alpha_A*(pushedChosen[i] - p_push)   
       p_yell = p_yell + alpha_C*(yellowChosen[i] - p_yell)
       
       # Calculating the Standard Expected Value
       EV_push = p_push*winAmtPushable[i]
       EV_pull = (1 - p_push)*(100 - winAmtPushable[i])
       EV_yell = p_yell*winAmtYellow[i]
       EV_blue = (1 - p_yell)*(100 - winAmtPushable[i])
       
       # Relative contribution of Action Value Learning verus Stimulus Value Learning
       EV_push_yell = weight*EV_push + (1 - weight)*EV_yell
       EV_push_blue = weight*EV_push + (1 - weight)*EV_blue
       EV_pull_yell = weight*EV_pull + (1 - weight)*EV_yell
       EV_pull_blue = weight*EV_pull + (1 - weight)*EV_blue
       
       # Calculating the soft-max function ovwer weightening Action and Color conditions
       if (pushedChosen[i] == 1 & yellowChosen[i] == 1):
           soft_max_EV[i] = exp(bet*EV_push_yell)/(exp(bet*EV_push_yell) + exp(bet*EV_pull_blue))
       elif (pushedChosen[i] == 1 & yellowChosen[i] == 0):
           soft_max_EV[i] = exp(bet*EV_push_blue)/(exp(bet*EV_push_blue) + exp(bet*EV_pull_yell))
       elif (pushedChosen[i] == 0 & yellowChosen[i] == 1):
           soft_max_EV[i] = exp(bet*EV_pull_yell)/(exp(bet*EV_pull_yell) + exp(bet*EV_push_blue))
       else (pushedChosen[i] == 0 & yellowChosen[i] == 0):
           soft_max_EV[i] = exp(bet*EV_pull_blue)/(exp(bet*EV_pull_blue) + exp(bet*EV_push_yell))      
    }   
}
model {
    /* learning rate parameters prior */
    alpha_A ~ beta(1,1); 
    alpha_A ~ beta(1,1); 

    /* Wieghtening parameter prior */
    weight ~ beta(1,1); 
    
    /* sensitivity parameter prior */
    bet ~ normal(0,2) T[0, 3];     
    
    /* RL likelihood */
    for (i in 1:N) { 
        acc[i] ~ bernoulli(soft_max_EV[i]);
    }
}
generated quantities { 
   vector[N] log_lik;  
   
    /*  RL Log density likelihood */
    for (i in 1:N) {
         log_lik[i] = bernoulli(soft_max_EV[i])
   }
}