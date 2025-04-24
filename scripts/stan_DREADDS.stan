//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

data {
    // overall structure
    int<lower=1> n_trials_overall;
    int<lower=1> n_subjects;
    int<lower=1> n_treatments;
    int<lower=1> n_actions;
    
    // trial-level structure
    array[n_trials_overall] int<lower=1, upper=n_subjects> id;
    array[n_trials_overall] int<lower=1, upper=n_treatments> treatment;
    array[n_trials_overall] int<lower=1, upper=n_actions> actions;
    array[n_trials_overall] int<lower=0, upper=1> rewards;
    
    // session boundaries
    array[n_trials_overall] int<lower=0, upper=1> new_session_flag;
}

parameters {
    // parameters are model unconstrained and later on brought back
    // into the desired scale
    
    // population level parameters
    // learning rate mean
    vector[n_treatments] mu_logit_alpha;
    // temperature mean
    vector[n_treatments] mu_log_tau;
    // learning rate deviations
    real<lower=0> sigma_alpha;
    real<lower=0> sigma_tau;
    
    // subject level parameters
    // these are deviations from the population parameters
    // effectively the random effects
    vector[n_subjects] z_alpha;
    vector[n_subjects] z_tau;
}

transformed parameters{
    // subject level parameters to their bounded scales
    vector<lower=0, upper=1>[n_trials_overall] alpha;
    vector<lower=0>[n_trials_overall] tau;
    
    for (t in 1:n_trials_overall) {
        int s = id[t];
        int trt = treatment[s];
        
        real logit_alpha_t = mu_logit_alpha[trt] + sigma_alpha * z_alpha[s];
        real log_tau_t    = mu_log_tau[trt] + sigma_tau * z_tau[s];
        
        alpha[t] = inv_logit(logit_alpha_t);
        tau[t] = exp(log_tau_t);
    }
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
    matrix[n_subjects, n_actions] Q_id;
    
    // priors
    mu_logit_alpha ~ normal(0, 1.5);
    mu_log_tau     ~ normal(0, 1.5);
    sigma_alpha    ~ exponential(1);
    sigma_tau      ~ exponential(1);
    z_alpha        ~ std_normal();
    z_tau          ~ std_normal();
    
    // initialize Q values
    for (s in 1:n_subjects) {
        Q_id[s, :] = rep_row_vector(0.5, n_actions);
    }
    
    // trial loop
    for (t in 1:n_trials_overall) {
        int id_idx = id[t];
        int act = actions[t];
        real rew = rewards[t];
        
        // trial specific parameters
        real current_alpha = alpha[t];
        real current_tau   = tau[t];
        
        // session reset logic
       // if (new_session_flag[t] == 1) {
       //     Q_id[id, :] = rep_row_vector(0.5, n_actions)
       // }
        
        // likelihood calculations
        vector[n_actions] Q_current = Q_id[id_idx]';
        vector[n_actions] actions_probs = softmax(Q_current / current_tau);
        act ~ categorical(actions_probs);
        
        // Q value update
        real prediction_error = rew - Q_current[act];
        Q_id[id_idx, act] = Q_current[act] + current_alpha * prediction_error;
    }
}

