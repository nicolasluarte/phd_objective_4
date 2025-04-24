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

ld <- read_csv("../datasets/lickometer_complete.csv")

### reinforcement learning ----

# read the data of fitted values slice the best fit
# DO NOT AVERAGE OUT !!
# filter out likelihoods of exactly 0 these are optimization errors
optimal_mdl_fit <- read_rds("../datasets/RL_model_fits.rds") %>% 
#    filter(likelihood != 0) %>% 
    group_by(ID, treatment) %>% 
    slice(which.min(likelihood)) %>% 
    mutate(
        drug = str_extract(treatment, "veh|tcs"),
        entropy_level = str_extract(treatment, "[a-z]+_[a-z]+")
    ) 
optimal_mdl_fit

# individual choice data to compute reward-side entropy
binary_entropy <- function(p) {
    if (p == 0 || p == 1) {
        return(0)
    }
    return(- (p * log2(p) + (1 - p) * log2(1 - p)))
}

ind_dat <- read_csv("../datasets/individual_choice_data.csv") %>% 
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
        temperature = ((opt_tau/5)* (n() - 1) + 0.5) / n()
    ) %>% 
    left_join(., ind_dat, by = c("ID", "treatment"))


# used beta to inform the model of hard boundaries impose in the
# optimization procedure
# could not compute alpha for 574
diff_rl_mdl_data <- rl_mdl_data %>% 
    ungroup() %>% 
    group_by(ID, entropy_level) %>% 
    mutate(
        temperature = (temperature - temperature[drug=="tcs"]),
        learning_rate = (learning_rate - learning_rate[drug=="tcs"])
    ) %>% 
    filter(drug=="veh")
diff_rl_mdl_data

temperature_mdl <- lme4::lmer(
    data = diff_rl_mdl_data,
    temperature ~ entropy_level + (1 | ID),
    control = lme4::lmerControl(
        optimizer = "bobyqa",
        optCtrl = list(maxfun = 2e5)
    )
)
summary(temperature_mdl)

temperature_mdl_emm <- emmeans::emmeans(
    temperature_mdl,
    specs = ~ entropy_level
) %>% broom.mixed::tidy(conf.int = TRUE)
temperature_mdl_emm

# same idea for the learning rate
lrate_mdl <- lme4::lmer(
    data = diff_rl_mdl_data,
    learning_rate ~ entropy_level + (1 | ID),
    control = lme4::lmerControl(
        optimizer = "bobyqa",
        optCtrl = list(maxfun = 2e5)
    )
)
summary(lrate_mdl)

lrate_mdl_emm <- emmeans::emmeans(
    lrate_mdl,
    ~ entropy_level,
    type = "response"
) %>% broom.mixed::tidy(conf.int = TRUE)
lrate_mdl_emm

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

# entropy explains changes in exploration
# and this is modulated by orexin

H_mdl <- glmmTMB::glmmTMB(
    data = rl_mdl_data %>% filter(entropy_level != "low_entropy"),
    temperature ~ drug * H+ entropy_level + (H + drug || ID),
    family = glmmTMB::beta_family(link = "logit")
)
summary(H_mdl)

H_mdl_emm <- emmeans::emmeans(
    H_mdl,
    pairwise~ H | drug | entropy_level,
    type = "response",
    at = list(H = seq(0, 1, 0.1))
)$contrasts %>% broom.mixed::tidy(conf.int = TRUE)
H_mdl_emm

H_mdl_emm %>%
    ggplot(aes(H, response)) +
    geom_line(aes(group = drug, color = drug)) +
    facet_wrap(~entropy_level)

H_mdl_emtrend <- emmeans::emtrends(
    H_mdl,
    pairwise ~ drug | H,
    var = "H",
    type = "response",
    at = list(H = seq(0, 1, 0.25))
)
H_mdl_emtrend

# p::tau ----

taup <- temperature_mdl_emm %>% 
    mutate(entropy_level = factor(as.factor(entropy_level),
                                  levels = c("low_entropy", "mid_entropy", "high_entropy"))) %>% 
    ggplot(aes(
        entropy_level, estimate
    )) +
    geom_pointrange(aes(ymin=conf.low, ymax=conf.high),
                    size = 1.25) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_point(aes(entropy_level, temperature),
            data = diff_rl_mdl_data,
            fill = "orange"
    ) +
    annotate("text", label="*", x=3, y=1.2, size=12) +
    scale_x_discrete(labels = c(
        latex2exp::TeX(r"($H_{low}$)"),
        latex2exp::TeX(r"($H_{mid}$)"),
        latex2exp::TeX(r"($H_{high}$)")
    )) +
    theme_uncertainty + 
    scale_y_continuous(breaks = seq(-1.5, 1.5, 0.5), 
                       limits = c(-1.5, 1.5), 
                       expand = c(0,0)) +
    ylab(
        latex2exp::TeX(r"($\hat{\tau}_{veh - tcs}$)")
    ) +
    xlab("")
taup

# p::alpha ----
alphap <- lrate_mdl_emm %>% 
    mutate(entropy_level = factor(as.factor(entropy_level),
                                  levels = c("low_entropy", "mid_entropy", "high_entropy"))) %>% 
    ggplot(aes(
        entropy_level, estimate
    )) +
    geom_pointrange(aes(ymin=conf.low, ymax=conf.high),
                    size = 1.25) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_point(aes(entropy_level, temperature),
            data = diff_rl_mdl_data,
            fill = "orange"
    ) +
    scale_x_discrete(labels = c(
        latex2exp::TeX(r"($H_{low}$)"),
        latex2exp::TeX(r"($H_{mid}$)"),
        latex2exp::TeX(r"($H_{high}$)")
    )) +
    theme_uncertainty + 
    scale_y_continuous(breaks = seq(-1.5, 1.5, 0.5), 
                       limits = c(-1.5, 1.5), 
                       expand = c(0,0)) +
    ylab(
        latex2exp::TeX(r"($\hat{\alpha}_{veh - tcs}$)")
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
tau_H <- H_mdl_emm %>% 
    ggplot(aes(
        H, estimate
    )) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_ribbon(aes(ymin=conf.low, ymax=conf.high),
                alpha = 0.05) +
    geom_line() +
    scale_x_continuous(breaks = seq(0, 1, 0.25)) +
    theme_uncertainty + 
    scale_y_continuous(breaks = seq(-1.5, 1.5, 0.5), 
                       limits = c(-1.5, 1.5), 
                       expand = c(0,0)) +
    ylab(
        latex2exp::TeX(r"($(\hat{\tau} | \bar{\alpha})_{veh-tcs}$)")
    ) +
    xlab(
        latex2exp::TeX(r"($H_{rewards}$)")
    )
tau_H

# figure ----

taup + alphap + tau_alpha_p + tau_H +
    plot_layout(widths = c(1, 1, 2, 2)) +
    plot_annotation(tag_levels = c("A"))
