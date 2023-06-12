/*  Logistic Regression Model over responce choise in behavioral data.
*/ 
data {
    int<lower=0> N;                  // number of data items
    int<lower=0> K;                  // number of coefficients (1 + n predictors)
    matrix[N, K] X;                  // predictor matrix
    int<lower=0, upper=1> y[N];      // outcome
}
parameters {
    vector[K] beta;       // regression coefficients
}
model {
    beta ~ cauchy(0, 5);  // prior coefficients
    y ~ bernoulli_logit(X * beta);  // likelihood
}
generated quantities {
    vector[N] log_likelihood;
    vector[N] y_hat;
    for (i in 1:N) {
        log_likelihood[i] = bernoulli_logit_lpmf(y[i] | X[i]*beta);   // log likelihood of each data point y[i] and each samples from beta
        y_hat[i] = bernoulli_logit_rng(X[i]*beta);   // generated data for each sample p
    }
}