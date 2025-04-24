# lib load ----
pacman::p_load(
    tidyverse,
    ggplot2,
    patchwork
)
setwd(this.path::here())

# Plot stuff ----
theme_uncertainty <- ggpubr::theme_pubr() +
    update_geom_defaults("point", list(size = 5, alpha = 0.5, shape = 21)) +
    theme(
        text = element_text(size = 24),
        axis.text=element_text(size=14),
        plot.margin = unit(c(0.5,0.5,0.5,0.5), "cm"),
        legend.position = "none"
    )

boxplot_sig_bracket <- function(group1, group2){
    ggsignif::geom_signif(
        comparisons = list(c(group1, group2)),
        map_signif_level = TRUE,
        textsize = 0,
        tip_length = 0
    )
}

# Rewards ----
# devtools::source_url("https://github.com/lab-cpl/lickometer-library/blob/main/src/lickometer_functions_compilate.R?raw=TRUE")

ld <- read_csv("../datasets/lickometer_complete_DREADDS.csv")

### reinforcement learning ----

# read the data of fitted values
optimal_mdl_fit <- read_rds("../datasets/RL_model_fits_DREADDS.rds") %>% 
    mutate(
        drug = replace_na(str_extract(treatment, "veh|cno"), "no_drug"),
        entropy_level = replace_na(str_extract(treatment, "[a-z]+_[a-z]+"), "low_entropy")
    ) %>% 
    group_by(ID, drug, entropy_level, session) %>%
    slice(which.min(likelihood)) %>% 
    mutate(vir = if_else(ID %in% c(809, 810, 811), "trt", "ctrl"))
optimal_mdl_fit


# individual choice data to compute reward-side entropy
binary_entropy <- function(p) {
    if (p == 0 || p == 1) {
        return(0)
    }
    return(- (p * log2(p) + (1 - p) * log2(1 - p)))
}

ind_dat <- read_csv("../datasets/individual_choice_data_DREADDS.csv") %>% 
    ungroup() %>% 
    group_by(ID, treatment) %>% 
    summarise(
        H = binary_entropy(mean(rewards))
    )
ind_dat
    
# this is to set everything to the same scale
# this makes sense because parameters are derived from a bounded
# optimization process
rl_mdl_data <- optimal_mdl_fit %>% 
    ungroup() %>% 
    mutate(
        learning_rate = (opt_alpha * (n() - 1) + 0.5) / n(),
        temperature = ((opt_tau/5)* (n() - 1) + 0.5) / n(),
        num_ent = as.numeric(factor(as.factor(entropy_level), levels = c("low_entropy", "mid_entropy", "high_entropy")))
    ) %>% 
    left_join(., ind_dat, by = c("ID", "treatment"))


# temperature plot is to make the "raw" parameters fit within model estimates
# subtraction is needed due to 3 animals per group, otherwise this would need
# triple interaction and 2 random slopes
# transformation due to heavy tails and to keep thing in the parameter estimation space
diff_rl_mdl_data <- rl_mdl_data %>% 
    ungroup() %>% 
    filter(drug != "no_drug") %>% 
    group_by(ID, entropy_level, vir) %>% 
    mutate(
        temperature_plt = (temperature - temperature[drug=="cno"])/temperature[drug=="cno"],
        temperature_plt = 1/(1+exp(-temperature_plt)),
        temperature = (temperature - temperature[drug=="cno"]),
        temperature = 1 / (1 + exp(-temperature)),
        learning_rate_plt = (learning_rate - learning_rate[drug=="cno"])/learning_rate[drug=="cno"],
        learning_rate_plt = 1/(1+exp(-learning_rate_plt)),
        learning_rate = (learning_rate - learning_rate[drug=="cno"]),
        learning_rate = 1 / (1 + exp(-learning_rate)),
    ) %>% 
    filter(drug=="veh")
diff_rl_mdl_data

## temperature mdl----
# entropy level makes this act as the relative difference within ID,
# otherwise this would be computing absolute differences in the parameter
# uncorrelated random slope from intercept, no convergence otherwise due to low sample size
temperature_mdl <- glmmTMB::glmmTMB(
    data = diff_rl_mdl_data,
    temperature ~ entropy_level * vir + (1 + entropy_level || ID),
    family = glmmTMB::beta_family(link = "logit")
)
summary(temperature_mdl)


# now we can make comparisons against 0 to keep things more interpretable
temperature_mdl_emm <- emmeans::emmeans(
    temperature_mdl,
    specs = ~ entropy_level | vir,
    type = "response"
) %>% emmeans::test()
temperature_mdl_emm

# to get confidence intervals
temperature_mdl_plt <- emmeans::emmeans(
    temperature_mdl,
    specs = ~ entropy_level | vir,
    type = "response"
) %>% broom.mixed::tidy(conf.int=TRUE)
temperature_mdl_plt

## learning rate mdl ----
lrate_mdl <- glmmTMB::glmmTMB(
    data = diff_rl_mdl_data,
    learning_rate ~ entropy_level * vir + (1 + entropy_level || ID),
    family = glmmTMB::beta_family(link = "logit")
)
summary(lrate_mdl)

lrate_mdl_emm <- emmeans::emmeans(
    lrate_mdl,
    specs = ~ entropy_level | vir,
    type = "response"
) %>% emmeans::test()
lrate_mdl_emm

lrate_mdl_plt <- emmeans::emmeans(
    lrate_mdl,
    specs = ~ entropy_level | vir,
    type = "response"
) %>% broom.mixed::tidy(conf.int = TRUE)
lrate_mdl_plt


### tau ~ alpha ----
tau_alpha_mdl <- glmmTMB::glmmTMB(
    data = rl_mdl_data %>% mutate(entropy_level = as.numeric(as.factor(entropy_level))),
    temperature ~ learning_rate + entropy_level,
    family = glmmTMB::beta_family(link="logit")
)
summary(tau_alpha_mdl)

tau_alpha_emm <- emmeans::emmeans(
    tau_alpha_mdl,
    ~ learning_rate + entropy_level,
    at = list(learning_rate = seq(0, 1, 0.1))
) %>% broom.mixed::tidy(conf.int = TRUE)
tau_alpha_emm

tau_alpha_emtrend <- emmeans::emtrends(
    tau_alpha_mdl,
    ~ learning_rate + entropy_level,
    var = "learning_rate"
) %>% broom.mixed::tidy(conf.int = TRUE)
tau_alpha_emtrend

## tau ~ H mdl ----

H_mdl <- glmmTMB::glmmTMB(
    data = diff_rl_mdl_data,
    temperature ~ H * vir + (1 | ID),
    family = glmmTMB::beta_family(link = "logit")
)
summary(H_mdl)

H_mdl_emm <- emmeans::emmeans(
    H_mdl,
    ~ H | vir,
    at = list(H = seq(0, 1, 0.1)),
) %>% broom.mixed::tidy(conf.int = TRUE)
H_mdl_emm

H_mdl_emtrend <- emmeans::emtrends(
    H_mdl,
    pairwise ~ H * vir,
    var = "H"
) %>% emmeans::test()
H_mdl_emtrend

# p::tau ----

# omitir la estrellita pk ahora agregue los dos grupos en un mismo plot
taup <- temperature_mdl_plt %>% 
    mutate(entropy_level = factor(as.factor(entropy_level),
                                  levels = c("low_entropy", "mid_entropy", "high_entropy"))) %>% 
    ggplot(aes(
        entropy_level, response
    )) +
    geom_pointrange(aes(ymin=conf.low, ymax=conf.high),
                    size = 1.25) +
    geom_hline(yintercept = 0.5, linetype = "dashed") +
    geom_point(aes(entropy_level, temperature_plt),
            data = diff_rl_mdl_data,
            fill = "orange"
    ) +
    annotate("text", label="*", x=3, y=0.85, size=12) +
    scale_x_discrete(labels = c(
        latex2exp::TeX(r"($H_{low}$)"),
        latex2exp::TeX(r"($H_{mid}$)"),
        latex2exp::TeX(r"($H_{high}$)")
    )) +
    theme_uncertainty + 
    scale_y_continuous(breaks = seq(0, 1, 0.25), 
                       limits = c(0, 1), 
                       expand = c(0,0)) +
    ylab(
        latex2exp::TeX(r"($\hat{\tau}_{veh - cno}$)")
    ) +
    xlab("") +
    facet_wrap(~vir)
taup

# p::alpha ----
alphap <- lrate_mdl_plt %>% 
    mutate(entropy_level = factor(as.factor(entropy_level),
                                  levels = c("low_entropy", "mid_entropy", "high_entropy"))) %>% 
    ggplot(aes(
        entropy_level, response
    )) +
    geom_pointrange(aes(ymin=conf.low, ymax=conf.high),
                    size = 1.25) +
    geom_hline(yintercept = 0.5, linetype = "dashed") +
    geom_point(aes(entropy_level, temperature_plt),
            data = diff_rl_mdl_data %>% filter(vir == "trt"),
            fill = "orange"
    ) +
    scale_x_discrete(labels = c(
        latex2exp::TeX(r"($H_{low}$)"),
        latex2exp::TeX(r"($H_{mid}$)"),
        latex2exp::TeX(r"($H_{high}$)")
    )) +
    theme_uncertainty + 
    scale_y_continuous(breaks = seq(0, 1, 0.5), 
                       limits = c(0, 1), 
                       expand = c(0,0)) +
    ylab(
        latex2exp::TeX(r"($\hat{\alpha}_{veh - cno}$)")
    ) +
    xlab("")
alphap

# p::tau by alpha ----
tau_alpha_p <- tau_alpha_emm %>% 
    ggplot(aes(
        learning_rate, estimate
    )) +
    geom_ribbon(aes(ymin=conf.low, ymax=conf.high),
                alpha = 0.05) +
    geom_line() +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_x_continuous(breaks = seq(0, 1, 0.25)) +
    theme_uncertainty + 
    scale_y_continuous(breaks = seq(-2.5, 2.5, 0.5), 
                       limits = c(-2.5, 2.5), 
                       expand = c(0,0)) +
    ylab(
        latex2exp::TeX(r"($\hat{\tau}$)")
    ) +
    xlab(
        latex2exp::TeX(r"($\alpha$)")
    )
tau_alpha_p

# p::tau given H ----
# mostly meaningless slopes are different but H itself has no significance (at no point)
tau_H <- H_mdl_emm %>% 
    ggplot(aes(
        H, estimate, group = vir
    )) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_ribbon(aes(ymin=conf.low, ymax=conf.high, fill = vir),
                alpha = 0.15) +
    geom_line() +
    scale_x_continuous(breaks = seq(0, 1, 0.25)) +
    theme_uncertainty + 
    scale_y_continuous(breaks = seq(-1.5, 1.5, 0.5), 
                       limits = c(-1.5, 1.5), 
                       expand = c(0,0)) +
    ylab(
        latex2exp::TeX(r"($\hat{\tau}_{veh-cno}$)")
    ) +
    xlab(
        latex2exp::TeX(r"($H_{rewards}$)")
    ) +
    scale_fill_manual(values = c("black", "orange"))
tau_H

# figure ----

taup + alphap + tau_alpha_p + tau_H +
    plot_layout(widths = c(1, 1, 2, 2)) +
    plot_annotation(tag_levels = c("A"))
