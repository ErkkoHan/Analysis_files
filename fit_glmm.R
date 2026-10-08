#!/usr/bin/env Rscript
suppressMessages(library(glmmTMB))

args <- commandArgs(trailingOnly = TRUE)
in_csv  <- if (length(args) >= 1) args[1] else "output/glmm_input.csv"
out_dir <- if (length(args) >= 2) args[2] else "output"
d <- read.csv(in_csv, fileEncoding = "UTF-8")

# ---- Factors and reporting units ------------------------------------------ #
d$season   <- relevel(factor(d$season, levels = c("Winter", "Spring", "Summer", "Autumn")),
                      ref = "Winter")
d$region   <- relevel(factor(d$region), ref = "Taipei")
d$business <- relevel(factor(d$business), ref = "ConveyorSushi")
d$building <- relevel(factor(d$building), ref = "InMall")
d$cid      <- factor(d$cid)
d$density_u10      <- d$density * 10
d$distribution_u10 <- d$distribution * 10

ITEMS_ALL <- c(paste0("facilities", 1:8), paste0("sanitation", 1:9))
ITEMS <- ITEMS_ALL[sapply(ITEMS_ALL, function(v) isTRUE(sd(d[[v]], na.rm = TRUE) > 0))]
RHS <- paste(c("density_u10", "distribution_u10", "season", "region", "business",
               "building", "temp", "humidity", ITEMS, "(1 | cid)"), collapse = " + ")

# ---- Models ----------------------------------------------------------------- #
m_cap <- glmmTMB(as.formula(paste("y_density ~", RHS)),
                 family = tweedie(link = "log"), data = d)
d$y_neg <- d$y_all_n - d$y_catch_n
m_pos <- glmmTMB(as.formula(paste("cbind(y_catch_n, y_neg) ~", RHS)),
                 family = binomial(link = "logit"), data = d)

# ---- Fixed effects ---------------------------------------------------------- #
tidy_fixed <- function(m) {
  s  <- summary(m)$coefficients$cond
  df <- data.frame(term = rownames(s), coef = s[, "Estimate"], se = s[, "Std. Error"],
                   p = s[, "Pr(>|z|)"], row.names = NULL)
  df$eff <- exp(df$coef)
  df$lo  <- exp(df$coef - 1.959964 * df$se)
  df$hi  <- exp(df$coef + 1.959964 * df$se)
  df
}
cap <- tidy_fixed(m_cap)
pos <- tidy_fixed(m_pos)
write.csv(cap, file.path(out_dir, "glmm_capture.csv"), row.names = FALSE)
write.csv(pos, file.path(out_dir, "glmm_posrate.csv"), row.names = FALSE)

# ---- Random-intercept SD and latent-scale ICC (Nakagawa et al. 2017) --------- #
sig <- function(m) sqrt(as.numeric(VarCorr(m)$cond$cid[1]))
icc_logit <- function(s2) s2 / (s2 + pi^2 / 3)
icc_log   <- function(s2, mu, phi, p) s2 / (s2 + log1p(phi * mu^(p - 2)))

s_cap   <- sig(m_cap)
s_pos   <- sig(m_pos)
phi_cap <- sigma(m_cap)
p_cap   <- family_params(m_cap)[[1]]
mu_cap  <- mean(fitted(m_cap))

meta <- data.frame(
  model     = c("capture_tweedie_glmm", "posrate_binomial_glmm"),
  n_obs     = c(nobs(m_cap), nobs(m_pos)),
  n_group   = c(nlevels(d$cid), nlevels(d$cid)),
  n_fixed   = c(nrow(cap), nrow(pos)),
  n_items   = c(length(ITEMS), length(ITEMS)),
  sigma_u   = c(s_cap, s_pos),
  icc       = c(icc_log(s_cap^2, mu_cap, phi_cap, p_cap), icc_logit(s_pos^2)),
  tweedie_p = c(p_cap, NA),
  phi       = c(phi_cap, NA),
  logLik    = c(as.numeric(logLik(m_cap)), as.numeric(logLik(m_pos))),
  AIC       = c(AIC(m_cap), AIC(m_pos)),
  converged = c(m_cap$fit$convergence == 0, m_pos$fit$convergence == 0)
)
write.csv(meta, file.path(out_dir, "glmm_meta.csv"), row.names = FALSE)
print(meta)

# ---- Completion flag (read by analysis_pipeline.py) ------------------------ #
writeLines(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), file.path(out_dir, "glmm_done.txt"))
