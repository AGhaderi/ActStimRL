/* Model RL, smae learning rate for action value learning and color value leanring*/ 
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
    int<lower=1> nConds;   // Number of conditions, Action and Color value learning
    array[N] int<lower=1, upper=2> condition;   // 1 indecates Action value learning and 2 indicates Color value learning
    real<lower=0, upper=1> p_push_init;         // Initial value of reward probability for pushed responce
    real<lower=0, upper=1> p_yell_init;         // Initial value of reward probability for Color responce
 }
parameters {
    array[nConds] real<lower=0, upper=1>  alpha;     // Learning rate
    array[nConds] real<lower=0, upper=1>  weightAct;    // Wieghtening of Action Learning Value against to Color Learnig Value
    array[nConds] real<lower=0>  sensitivity;  // With a higher sensitivity value θ, choices are more sensitive to value differences
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
        EV_push_yell = weightAct[condition[i]]*EV_push + (1 - weightAct[condition[i]])*EV_yell;
        EV_push_blue = weightAct[condition[i]]*EV_push + (1 - weightAct[condition[i]])*EV_blue;
        EV_pull_yell = weightAct[condition[i]]*EV_pull + (1 - weightAct[condition[i]])*EV_yell;
        EV_pull_blue = weightAct[condition[i]]*EV_pull + (1 - weightAct[condition[i]])*EV_blue;
       
        /* Calculating the soft-max function over weightening Action and Color conditions*/ 
        // pushed and yellow vs pulled and blue
        if ((pushed[i] == 1 && yellowChosen[i] == 1) || (pushed[i] == 0 && yellowChosen[i] == 0))
            soft_max_EV[i] = exp(sensitivity[condition[i]]*EV_push_yell)/(exp(sensitivity[condition[i]]*EV_push_yell) + exp(sensitivity[condition[i]]*EV_pull_blue));

        // pushed and blue vs pulled and yellow
        if ((pushed[i] == 1 && yellowChosen[i] == 0) || (pushed[i] == 0 && yellowChosen[i] == 1))
            soft_max_EV[i] = exp(sensitivity[condition[i]]*EV_push_blue)/(exp(sensitivity[condition[i]]*EV_push_blue) + exp(sensitivity[condition[i]]*EV_pull_yell));  
          
        // RL rule update for computing prediction error and internal value expectation for the next trial based on the current reward output and interal value expectation
        if (pushed[i] == 1){
            p_push = p_push + alpha[condition[i]]*(rewarded[i] - p_push); 
            p_pull = 1 - p_push;
        }
        else{
            p_pull = p_pull + alpha[condition[i]]*(rewarded[i] - p_pull);
            p_push = 1 - p_pull;
        }    
        if (yellowChosen[i] == 1){
           p_yell = p_yell + alpha[condition[i]]*(rewarded[i] - p_yell);
           p_blue = 1 - p_yell;
        }    
        else{
           p_blue = p_blue + alpha[condition[i]]*(rewarded[i] - p_blue);
           p_yell = 1 - p_blue;           
        }
       
    }   
}
model {
    /* prior */
    for (c in 1:nConds){
        sensitivity[c] ~ gamma(1,1);
        alpha[c] ~ beta(1, 1); 
        weightAct[c] ~ beta(1, 1);
    }
    /* RL likelihood */
    for (i in 1:N) { 
        pushed[i] ~ bernoulli(soft_max_EV[i]);
    }
}