#libs ----
pacman::p_load(
    tidyverse,
    ggplot2,
    lme4,
    lmerTest,
    furrr,
    ggpubr,
    patchwork,
    lme4,
    lmerTest,
    zeallot,
    nloptr,
    robustlmm,
    cmdstanr,
    posterior
)
setwd(this.path::here())

# Plot stuff ----
theme_uncertainty <- ggpubr::theme_pubr() +
    update_geom_defaults("point", list(size = 5, alpha = 0.5, shape = 21)) +
    update_geom_defaults("boxplot", list(width=0.5)) +
    theme(
        text = element_text(size = 24),
        axis.text=element_text(size=14),
        plot.margin = unit(c(0.5,0.5,0.5,0.5), "cm"),
        legend.position = "none"
    )

# soft-max action selection function ----
soft_max <- function(Q_vector, tau){
    # we compute the log of probabilities to get a stable response
    # we do not need to log Q because log(exp(x)) = x
    # so note that that I did not apply exp to Q_exp
    # because later on if would log(exp(Q_scaled))
    # 1/tau because Im multiplying and I want to use temperature not inverse temp
    Q_scaled <- (as.vector(Q_vector) * (1/tau))
    # use the log rule for division
    # and do the log-sum-exp trick to avoid under and overflows
    soft_max_out <- Q_scaled - matrixStats::logSumExp(Q_scaled)
    return(soft_max_out)
}

# likelihood function ----
likelihood_function <- function(
        theta, # theta[1] learning rate, theta[2] temperature
        actions,
        rewards
        ){
    # parameters
    alpha <- theta[1]
    tau <- theta[2]
    # this vector will store the log-probabilities
    # this is a vector of all 0's
    logp_actions_t <- numeric(length(actions))
    
    # initialize the Q table, with total indifference or random
    random_init = FALSE
    if (random_init == TRUE){
       L = round(runif(1, min=0, max=1), 1) 
       R = round(runif(1, min=0, max=1), 1) 
       Q <- list(c(L, R))
    }
    else{
        Q <- c(0.5, 0.5)
    }
    
    # do simulations
    for (i in 1:length(actions)){
        # apply softmax but in log form
        soft_max_out <- soft_max(Q, tau)
        
        # a=1 left, a=2 right
        r <- rewards[i]
        a <- actions[i]
        # store log probability of each action
        logp_actions_t[i] <- soft_max_out[a]
        
        # update Q values
        Q[a] <- Q[a] + (alpha * (r - Q[a]))
    }
    return(-sum(logp_actions_t[-1]))
}

# example data generation ----
data_gen <- function(learning_rate, tau, n_trials=100){
    reward_probabilities <- c(0.5, 1)
    actions <- numeric(length(n_trials))
    rewards <- numeric(length(n_trials))
    Q_L <- numeric(length(n_trials))
    Q_R <- numeric(length(n_trials))
    Q <- c(0.5, 0.5)
    for (i in 1:n_trials){
        Q_exp <- exp(unlist(Q) / tau)
        prob_a <- Q_exp / sum(Q_exp)
        # I need the exp to get back normal probs
        a <- sample(c(1, 2), prob=prob_a, replace=TRUE, size=1)
        r <- sample(c(0, 1), prob=c(1-reward_probabilities[a], reward_probabilities[a]),
                    replace=TRUE,
                    size=1)
        Q[a] <- Q[a] + (learning_rate * (r - Q[a]))
        # store values
        actions[i] <- a
        rewards[i] <- r
        Q_L[i] <- Q[1]
        Q_R[i] <- Q[2]
    }
    return(tibble(
        actions = actions,
        rewards = rewards,
        Q_L = (Q_L),
        Q_R = (Q_R)
    ))
}

# parameter recovery ----

# simulate data and get log likelihood of true parameters
parameter_grid <- expand_grid(
    alpha = seq(0.01, 1, length.out = 10),
    tau = seq(0.01, 5, length.out = 10)
) %>% 
    mutate(s=row_number()) %>% 
    group_by(s) %>% 
    group_split() %>% 
    map(., function(X){
        dat_sim <- data_gen(X$alpha, X$tau, n_trials=100)
    lower_bounds <- c(alpha = 0.001, tau = 0.001) # Use small positive values
    upper_bounds <- c(alpha = 1, tau = 5)      # Reasonable upper bounds
    start_par <- c(
        runif(1, min = lower_bounds[1], max = upper_bounds[1]),
        runif(1, min = lower_bounds[2], max = upper_bounds[2])
    )
    param_optim <- optim(
        par = start_par,
        fn = likelihood_function,
        method = "L-BFGS-B",
        lower = lower_bounds,
        upper = upper_bounds,
        actions = dat_sim$actions,
        rewards = dat_sim$rewards
    )
    optim_res <- param_optim$par
    return(tibble(
        true_alpha = X$alpha,
        true_tau = X$tau,
        opt_alpha = optim_res[1],
        opt_tau = optim_res[2]
    ))
    })

# optimization residuals
parameter_sweep_d <-bind_rows(parameter_grid) %>% 
    mutate(
        res_alpha = ((true_alpha) - (opt_alpha)),
        res_tau = ((true_tau) - (opt_tau)),
        distance = sqrt(((true_alpha-opt_alpha)/sd(true_alpha))^2 +
                            ((true_tau-opt_tau)/sd(true_tau))^2)
    )
parameter_sweep_d


param_sweep_p1 <- parameter_sweep_d %>% 
    ggplot(aes(
        scale(true_alpha), scale(true_tau)
    )) +
    geom_tile(aes(fill=distance)) +
    geom_text(aes(label=round(distance,1)), size=5) +
    coord_fixed(ratio = 1) +
    scale_y_continuous(breaks = c(-2,0,2), 
                       limits = c(-2,2), 
                       expand = c(0,0)) +
    scale_x_continuous(breaks = c(-2,0,2), 
                       limits = c(-2,2), 
                       expand = c(0,0)) +
    theme_uncertainty +
    theme(legend.position = "right",
          legend.title=element_text(size=12),
          legend.text = element_text(size=10)) +
    ylab(latex2exp::TeX(r"($\tau_{true}$)")) +
    xlab(latex2exp::TeX(r"($\alpha_{true}$)")) +
    labs(fill=latex2exp::TeX(r"($\norm{\cdot}_{2}$)")) +
    scale_fill_gradientn(breaks=seq(0,5,1), limits=c(0, 5),
                         colours = terrain.colors(3))
param_sweep_p1
    



# empirical data ----
devtools::source_url("https://github.com/lab-cpl/lickometer-library/blob/main/src/lickometer_functions_compilate.R?raw=TRUE")

tmp <- load_experiment(
    metadataFileName = "../datasets/metadata_DREADDS/metadata.csv",
    data_directory_path = "../datasets/lickometer_DREADDS"
)
write_csv(x = tmp, file = "../datasets/lickometer_complete_DREADDS.csv")

ed <- read_csv("../datasets/lickometer_complete_DREADDS.csv")

ed_choice <- ed %>% 
    group_by(ID, fecha) %>% 
    arrange(tiempo, .by_group = TRUE) %>% 
    ungroup() %>% 
    mutate(
        treatment = interaction(estimulo_spout_1, estimulo_spout_2, droga),
        treatment = case_when(
            treatment == "cm_100.cm_100.na_na_na_na" ~ "baseline",
            treatment == "cm_100.cm_100.veh_na_na_na" ~ "low_entropy_veh",
            treatment == "cm_100.cm_100.cno_na_na_na" ~ "low_entropy_cno",
            treatment == "cm_50.cm_100.veh_na_na_na" ~ "mid_entropy_veh",
            treatment == "cm_50.cm_100.cno_na_na_na" ~ "mid_entropy_cno",
            treatment == "cm_100.cm_50.veh_na_na_na" ~ "mid_entropy_veh",
            treatment == "cm_100.cm_50.cno_na_na_na" ~ "mid_entropy_cno",
            treatment == "cm_25.cm_50.veh_na_na_na" ~ "high_entropy_veh",
            treatment == "cm_25.cm_50.cno_na_na_na" ~ "high_entropy_cno",
            treatment == "cm_50.cm_25.veh_na_na_na" ~ "high_entropy_veh",
            treatment == "cm_50.cm_25.cno_na_na_na" ~ "high_entropy_cno",
            TRUE ~ "ERROR"
        )
    ) %>% 
    filter(actividad != -1, treatment != "baseline") %>% 
    group_by(ID, treatment, sensor) %>% 
    mutate(
        new_event = if_else(evento!=lag(evento) & evento > 0, 1, 0),
        new_reward = if_else(exito!=lag(exito), 1, 0)
    ) %>%
    ungroup() %>% 
    filter(new_event==1) %>% 
    group_by(ID, treatment, evento) %>% 
    mutate(
        actions = sensor+1,
        rewards = max(new_reward, na.rm = TRUE)
    ) %>% 
    ungroup() %>% 
    select(ID, actions, rewards, treatment) %>% 
    group_by(ID, treatment) %>% 
    drop_na() %>% 
    group_split()
write_csv(x = ed_choice %>% bind_rows(), "../datasets/individual_choice_data_DREADDS.csv")

# model fit ----
# do at least 1000 runs
plan(multisession, workers = 12)
model_fits <- 1:1000 %>% 
    future_map_dfr(., function(iteration){
        it <- map_dfr(ed_choice, function(X){
            dat_sim <- X
            lower_bounds <- c(alpha = 0.001, tau = 0.001) # Use small positive values
            upper_bounds <- c(alpha = 1, tau = 5)      # Reasonable upper bounds
            start_par <- c(
                runif(1, min = lower_bounds[1], max = upper_bounds[1]),
                runif(1, min = lower_bounds[2], max = upper_bounds[2])
            )
        param_optim <- optim(
            par = start_par,
            fn = likelihood_function,
            method = "L-BFGS-B",
            lower = lower_bounds,
            upper = upper_bounds,
            actions = dat_sim$actions,
            rewards = dat_sim$rewards
        )
        optim_res <- param_optim$par
        return(tibble(
            opt_alpha = optim_res[1],
            opt_tau = optim_res[2],
            likelihood = param_optim$value,
            iteration = iteration,
            ID = dat_sim$ID[1],
            treatment = dat_sim$treatment[1]
        ))
        })
        return(it)
    }, .options = furrr_options(seed = 6911))
write_rds(model_fits, "../datasets/RL_model_fits_DREADDS.rds")
