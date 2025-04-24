# libs ----
pacman::p_load(
    tidyverse,
    ggplot2,
    posterior,
    bayesplot,
    cmdstanr,
    tidybayes
)
setwd(this.path::here())

# CNO model ----
cno_model_data <- read_csv("../datasets/individual_choice_data_complete_DREADDS.csv") %>% 
    mutate(
        id = as.numeric(as.factor(ID)),
        treatment = as.numeric(factor(as.factor(treatment), levels = 
                                          c(
                                              "baseline",
                                              "low_entropy_veh",
                                              "mid_entropy_veh",
                                              "high_entropy_veh",
                                              "low_entropy_cno",
                                              "mid_entropy_cno",
                                              "high_entropy_cno"
                                          ))),
        session = as.numeric(as.factor(n_sesion))
    ) %>% 
    select(id, session, treatment, actions, rewards) %>% 
    group_by(id, session) %>% 
    mutate(trial_n = row_number()) %>% 
    ungroup() %>% 
    group_by(id, session, trial_n) %>% 
    arrange(session, .by_group = TRUE) %>% 
    ungroup() %>% 
    mutate(n_trials_overall = row_number(),
           new_session_flag = replace_na(as.numeric(session!=lag(session)), 0))
cno_model_data

# pass vars to stan model ----
# need to add virus hcrt or control
n_trials_overall <- max(cno_model_data$n_trials_overall)
n_subjects       <- length(unique(cno_model_data$id))
n_treatments     <- length(unique(cno_model_data$treatment))
n_actions        <- length(unique(cno_model_data$actions))
new_session_flag <- cno_model_data$new_session_flag
id               <- cno_model_data$id
treatment        <- cno_model_data$treatment
actions          <- cno_model_data$actions
rewards          <- cno_model_data$rewards


stan_data <- list(
    n_trials_overall = n_trials_overall,
    n_subjects       = n_subjects,
    n_treatments     = n_treatments,
    n_actions        = n_actions,
    id               = id,
    treatment        = treatment,
    actions          = actions,
    rewards          = rewards,
    new_session_flag = new_session_flag
)

# compile model ----

mdl <- cmdstan_model("../scripts/stan_DREADDS.stan",
                     cpp_options = list(stan_threads = TRUE))

# run the model ----

fit <- mdl$sample(
    data = stan_data,
    seed = 42,
    chains = 4,
    parallel_chains = 12,
    threads_per_chain = 1,
    iter_warmup = 1000,
    iter_sampling = 1500,
    refresh = 200,
    show_messages = TRUE,
    adapt_delta = 0.85
)

# get results ----
