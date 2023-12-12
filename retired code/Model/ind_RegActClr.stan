/* Regression Model over responce time in behavioral data.
*/ 
data {
    int<lower=0> N;                  // number of data items
    int<lower=0> K;                  // number of coefficients (1 + n predictors)
    matrix[N, K] X;                  // predictor matrix
    int<lower=0, upper=1> y[N];      // outcome
}
parameters {
    vector[K] beta;       // regression coefficients
    real<lower=0> sigma;  // error scale
}
model {
    beta ~ cauchy(0, 5);  // prior coefficients
    sigma ~ cauchy(0, 5);         // prior error term
    y ~ normal(X * beta, sigma);  // likelihood
}
generated quantities {
    vector[N] log_likelihood;
    vector[N] y_hat;
    for (i in 1:N) {
        log_likelihood[i] = normal_lpdf(y[i] | X[i]*beta, sigma);   // log likelihood of each data point y[i] and each samples from beta
        y_hat[i] = normal_rng(X[i]*beta, sigma);   // generated data for each sample p
    }
}