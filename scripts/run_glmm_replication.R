#!/usr/bin/env Rscript
# ==============================================================================
# GLMM Replication Models
#
# Runs the three legacy thesis model sets on the revamp full-sample dataset:
#   Model C: Group Characteristics (most data available, run first)
#   Model A: Policy Salience
#   Model B: Group-Politician Linkage
#   Empty Model: ICC decomposition (run last, lightweight)
#
# Input:  data/output/analysis_dataset_replication.csv
# Output: results_replication/glmm_replication_results.csv
#         results_replication/glmm_replication_comparison.csv
#         results_replication/glmm_replication_forest.png
#
# Usage: Rscript scripts/run_glmm_replication.R
#        source("scripts/run_glmm_replication.R")  # in RStudio
# ==============================================================================

# --- Setup -------------------------------------------------------------------

# Set working directory to project root
if (!grepl("ThesisPipelineRework", getwd())) {
  candidates <- c(
    file.path(Sys.getenv("USERPROFILE"), "OneDrive", "Desktop", "ThesisPipelineRework"),
    file.path(Sys.getenv("USERPROFILE"), "Desktop", "ThesisPipelineRework"),
    "."
  )
  for (path in candidates) {
    if (dir.exists(path)) { setwd(path); break }
  }
}
cat(sprintf("Working directory: %s\n", getwd()))
cat(sprintf("R version: %s\n", R.version.string))

cat("============================================================\n")
cat("GLMM REPLICATION ANALYSIS\n")
cat("============================================================\n\n")

# Ensure user library exists (system library may not be writable)
user_lib <- Sys.getenv("R_LIBS_USER")
if (nzchar(user_lib)) {
  dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(user_lib, .libPaths()))
}

# Install/load packages
required <- c("tidyverse", "lme4", "broom.mixed")
missing <- required[!sapply(required, requireNamespace, quietly = TRUE)]
if (length(missing) > 0) {
  cat("Installing missing packages:", paste(missing, collapse = ", "), "\n")
  install.packages(missing, repos = "https://cran.r-project.org", lib = user_lib)
}

suppressPackageStartupMessages({
  library(tidyverse)
  library(lme4)
  library(broom.mixed)
})

# Create output directory
dir.create("results_replication", showWarnings = FALSE)

# --- Helper Functions --------------------------------------------------------

fit_glmer_safe <- function(formula, data, model_name) {
  cat(sprintf("\n  Fitting %s...\n", model_name))
  cat(sprintf("  Rows: %s | Unique org_ids: %s | Unique issue_areas: %s\n",
              format(nrow(data), big.mark = ","),
              format(length(unique(data$org_id)), big.mark = ","),
              format(length(unique(data$issue_area)), big.mark = ",")))

  formula_str <- deparse(formula, width.cutoff = 500)

  # Strategy 1: glmer with nAGQ=0 and crossed random effects
  if (grepl("org_id.*issue_area|issue_area.*org_id", formula_str)) {
    cat("  Attempt 1: glmer nAGQ=0 with crossed random effects...\n")
    result <- tryCatch({
      model <- glmer(
        formula, data = data, family = binomial,
        nAGQ = 0,
        control = glmerControl(
          optimizer = "bobyqa",
          optCtrl = list(maxfun = 50000),
          calc.derivs = FALSE
        )
      )
      cat("  Success (crossed random effects).\n")
      list(success = TRUE, model = model, method = "glmer_crossed")
    }, error = function(e) {
      cat(sprintf("  Failed: %s\n", e$message))
      NULL
    })
    if (!is.null(result)) return(result)

    # Strategy 2: drop org_id, keep issue_area only (18 levels - very fast)
    cat("  Attempt 2: glmer with (1|issue_area) only...\n")
    formula_issue <- as.formula(gsub("\\+ *\\(1 *\\| *org_id\\)", "", formula_str))
    result <- tryCatch({
      model <- glmer(
        formula_issue, data = data, family = binomial,
        nAGQ = 0,
        control = glmerControl(
          optimizer = "bobyqa",
          optCtrl = list(maxfun = 50000),
          calc.derivs = FALSE
        )
      )
      cat("  Success (issue_area random effect only).\n")
      list(success = TRUE, model = model, method = "glmer_issue_only")
    }, error = function(e) {
      cat(sprintf("  Failed: %s\n", e$message))
      NULL
    })
    if (!is.null(result)) return(result)
  } else {
    # Formula doesn't have crossed effects - try as-is
    cat("  Attempt: glmer nAGQ=0...\n")
    result <- tryCatch({
      model <- glmer(
        formula, data = data, family = binomial,
        nAGQ = 0,
        control = glmerControl(
          optimizer = "bobyqa",
          optCtrl = list(maxfun = 50000),
          calc.derivs = FALSE
        )
      )
      cat("  Success.\n")
      list(success = TRUE, model = model, method = "glmer_nAGQ0")
    }, error = function(e) {
      cat(sprintf("  Failed: %s\n", e$message))
      NULL
    })
    if (!is.null(result)) return(result)
  }

  # Strategy 3: fixed effects glm (no random effects)
  cat("  Attempt 3: glm (fixed effects only)...\n")
  formula_fixed <- gsub("\\+ *\\(1 *\\| *[^)]+\\)", "", formula_str)
  formula_fixed <- as.formula(formula_fixed)
  result <- tryCatch({
    model <- glm(formula_fixed, data = data, family = binomial)
    cat("  Success (fixed effects only).\n")
    list(success = TRUE, model = model, method = "glm_fixed")
  }, error = function(e) {
    cat(sprintf("  glm also failed: %s\n", e$message))
    list(success = FALSE, error = e$message, method = "none")
  })

  return(result)
}

extract_coefficients <- function(result, model_label) {
  if (!result$success) return(tibble())

  coefs <- tryCatch({
    tidy(result$model, effects = "fixed", conf.int = TRUE) %>%
      mutate(
        odds_ratio = exp(estimate),
        or_lower = exp(conf.low),
        or_upper = exp(conf.high),
        sig = case_when(
          p.value < 0.001 ~ "***",
          p.value < 0.01 ~ "**",
          p.value < 0.05 ~ "*",
          TRUE ~ ""
        ),
        model = model_label,
        method = result$method
      )
  }, error = function(e) {
    cat(sprintf("  Could not extract coefficients: %s\n", e$message))
    tibble()
  })

  return(coefs)
}

# --- Load Data ---------------------------------------------------------------

cat("Loading data...\n")
df <- tryCatch({
  read.csv("data/output/analysis_dataset_replication.csv",
           stringsAsFactors = FALSE)
}, error = function(e) {
  cat(sprintf("FATAL: Cannot load data: %s\n", e$message))
  stop(e)
})

cat(sprintf("  Total rows: %s\n", format(nrow(df), big.mark = ",")))
cat(sprintf("  Columns: %d\n", ncol(df)))
cat(sprintf("  Unique orgs: %s\n", format(length(unique(df$org_id)), big.mark = ",")))
cat(sprintf("  Mention rows: %s\n", format(sum(df$is_zero_mention == 0, na.rm = TRUE), big.mark = ",")))
cat(sprintf("  Zero-mention rows: %s\n\n", format(sum(df$is_zero_mention == 1, na.rm = TRUE), big.mark = ",")))

# --- Data Diagnostics --------------------------------------------------------

cat("--- Data Diagnostics ---\n")
cat(sprintf("  prominence_prediction values: %s\n",
            paste(sort(unique(df$prominence_prediction[!is.na(df$prominence_prediction)])), collapse = ", ")))
cat(sprintf("  Rows with NA prominence: %d\n", sum(is.na(df$prominence_prediction))))

check_cols <- c("log_lobbying", "org_age", "salience_score", "terms_served_before",
                "bills_referenced", "policy_overlap")
for (col in check_cols) {
  if (col %in% names(df)) {
    vals <- suppressWarnings(as.numeric(df[[col]]))
    n_valid <- sum(!is.na(vals) & is.finite(vals))
    n_na <- sum(is.na(vals))
    cat(sprintf("  %s: %s valid, %d NA/invalid\n", col,
                format(n_valid, big.mark = ","), n_na))
  }
}

# --- Prepare Model Data ------------------------------------------------------

cat("\nPreparing model data...\n")

model_data <- df %>%
  filter(is_zero_mention == 0) %>%
  mutate(
    prominence = as.integer(prominence_prediction),
    org_id = as.factor(org_id),
    issue_area = as.factor(issue_area),
    is_democrat = as.integer(is_democrat),
    is_senate = as.integer(is_senate),
    is_labor = as.integer(is_labor),
    is_single_issue = as.integer(is_single_issue),
    is_trade = as.integer(is_trade),
    is_membership_org = as.integer(is_membership_org),
    log_lobbying = suppressWarnings(as.numeric(log_lobbying)),
    org_age = suppressWarnings(as.numeric(org_age)),
    policy_scope = as.integer(policy_scope),
    terms_served_before = suppressWarnings(as.numeric(terms_served_before)),
    up_for_reelection = as.integer(up_for_reelection),
    bills_referenced = as.integer(bills_referenced),
    policy_overlap = as.integer(policy_overlap),
    salience_score = suppressWarnings(as.numeric(salience_score)),
    salience_cat = factor(salience_category, levels = c("low", "medium", "high"))
  )

if ("bills_sponsored" %in% names(model_data)) {
  model_data$bills_sponsored <- suppressWarnings(as.numeric(model_data$bills_sponsored))
  model_data$log_bills_sponsored <- log1p(model_data$bills_sponsored)
}

# Drop unused factor levels to reduce memory in random effects
model_data$org_id <- droplevels(model_data$org_id)
model_data$issue_area <- droplevels(model_data$issue_area)

cat(sprintf("  Model data rows: %s\n", format(nrow(model_data), big.mark = ",")))
cat(sprintf("  Unique org_ids: %s\n", format(nlevels(model_data$org_id), big.mark = ",")))
cat(sprintf("  Unique issue_areas: %d\n", nlevels(model_data$issue_area)))
cat(sprintf("  Rows with salience: %s\n",
            format(sum(!is.na(model_data$salience_cat)), big.mark = ",")))
cat(sprintf("  Rows with seniority: %s\n",
            format(sum(!is.na(model_data$terms_served_before)), big.mark = ",")))

gc()

# Track results
model_results <- list()
all_coefs <- tibble()

# --- Model C: Group Characteristics (run first - most data available) --------

cat("\n--- Model C: Group Characteristics ---\n")
cat("prominence ~ org_age + log_lobbying + policy_scope\n")
cat("           + is_single_issue + is_labor + is_membership_org\n")
cat("           + (1|org_id) + (1|issue_area)\n")

model_c_data <- model_data %>%
  filter(!is.na(log_lobbying), !is.na(org_age), !is.na(policy_scope),
         !is.na(prominence))
model_c_data$org_id <- droplevels(model_c_data$org_id)
model_c_data$issue_area <- droplevels(model_c_data$issue_area)

cat(sprintf("  Rows: %s | org_ids: %s | issue_areas: %d\n",
            format(nrow(model_c_data), big.mark = ","),
            format(nlevels(model_c_data$org_id), big.mark = ","),
            nlevels(model_c_data$issue_area)))

result_c <- fit_glmer_safe(
  prominence ~ org_age + log_lobbying + policy_scope +
    is_single_issue + is_labor + is_membership_org +
    (1 | org_id) + (1 | issue_area),
  data = model_c_data,
  model_name = "Model C"
)

model_results[["C"]] <- result_c
mc_coef <- extract_coefficients(result_c, "C")
if (nrow(mc_coef) > 0) {
  all_coefs <- bind_rows(all_coefs, mc_coef)
  cat("\nModel C Coefficients:\n")
  mc_coef %>% select(term, estimate, std.error, odds_ratio, p.value, sig) %>% print(n = 20)
}

if (result_c$success) {
  saveRDS(result_c$model, "results_replication/model_c.rds")
  write_csv(mc_coef, "results_replication/model_c_coefficients.csv")
  cat("  Saved model C results.\n")
}

rm(model_c_data); gc()

# --- Model A: Policy Salience ------------------------------------------------

cat("\n--- Model A: Policy Salience ---\n")
cat("prominence ~ salience_cat + is_democrat + is_senate\n")
cat("           + is_membership_org + is_single_issue + is_labor\n")
cat("           + (1|org_id) + (1|issue_area)\n")

model_a_data <- model_data %>%
  filter(!is.na(salience_cat), !is.na(is_democrat), !is.na(is_senate),
         !is.na(prominence))
model_a_data$org_id <- droplevels(model_a_data$org_id)
model_a_data$issue_area <- droplevels(model_a_data$issue_area)

cat(sprintf("  Rows after NA removal: %s\n", format(nrow(model_a_data), big.mark = ",")))

if (nrow(model_a_data) < 100) {
  cat("  Insufficient data for Model A. Skipping.\n")
  result_a <- list(success = FALSE, error = "Insufficient data", method = "none")
} else {
  result_a <- fit_glmer_safe(
    prominence ~ salience_cat + is_democrat + is_senate +
      is_membership_org + is_single_issue + is_labor +
      (1 | org_id) + (1 | issue_area),
    data = model_a_data,
    model_name = "Model A"
  )
}

model_results[["A"]] <- result_a
ma_coef <- extract_coefficients(result_a, "A")
if (nrow(ma_coef) > 0) {
  all_coefs <- bind_rows(all_coefs, ma_coef)
  cat("\nModel A Coefficients:\n")
  ma_coef %>% select(term, estimate, std.error, odds_ratio, p.value, sig) %>% print(n = 20)
}

if (result_a$success) {
  saveRDS(result_a$model, "results_replication/model_a.rds")
  write_csv(ma_coef, "results_replication/model_a_coefficients.csv")
  cat("  Saved model A results.\n")
}

rm(model_a_data); gc()

# --- Model B: Group-Politician Linkage ----------------------------------------

cat("\n--- Model B: Group-Politician Linkage ---\n")

b_formula_parts <- c("prominence ~ terms_served_before + up_for_reelection",
                      "+ is_democrat + is_senate",
                      "+ is_single_issue + is_labor + is_membership_org",
                      "+ log_lobbying")

b_filter_cols <- c("is_democrat", "is_senate", "terms_served_before",
                    "up_for_reelection", "log_lobbying")

if (sum(!is.na(model_data$policy_overlap)) > 1000) {
  b_formula_parts <- c(b_formula_parts[1],
                        "+ policy_overlap",
                        b_formula_parts[-1])
  b_filter_cols <- c(b_filter_cols, "policy_overlap")
  cat("  Including policy_overlap\n")
} else {
  cat("  Excluding policy_overlap (insufficient data)\n")
}

if ("bills_sponsored" %in% names(model_data) &&
    sum(!is.na(model_data$bills_sponsored)) > 1000) {
  b_formula_parts <- c(b_formula_parts, "+ log_bills_sponsored")
  b_filter_cols <- c(b_filter_cols, "bills_sponsored")
  cat("  Including bills_sponsored\n")
} else {
  b_formula_parts <- c(b_formula_parts, "+ bills_referenced")
  b_filter_cols <- c(b_filter_cols, "bills_referenced")
  cat("  Including bills_referenced (speech-level proxy)\n")
}

b_formula_str <- paste(c(paste(b_formula_parts, collapse = " "),
                          "+ (1 | org_id) + (1 | issue_area)"),
                        collapse = " ")
cat("  Formula:", b_formula_str, "\n")

model_b_data <- model_data
for (col in b_filter_cols) {
  model_b_data <- model_b_data %>% filter(!is.na(.data[[col]]))
}
model_b_data <- model_b_data %>% filter(!is.na(prominence))
model_b_data$org_id <- droplevels(model_b_data$org_id)
model_b_data$issue_area <- droplevels(model_b_data$issue_area)

cat(sprintf("  Rows after NA removal: %s\n", format(nrow(model_b_data), big.mark = ",")))

if (nrow(model_b_data) < 100) {
  cat("  Insufficient data for Model B. Skipping.\n")
  result_b <- list(success = FALSE, error = "Insufficient data", method = "none")
} else {
  result_b <- fit_glmer_safe(
    as.formula(b_formula_str),
    data = model_b_data,
    model_name = "Model B"
  )
}

model_results[["B"]] <- result_b
mb_coef <- extract_coefficients(result_b, "B")
if (nrow(mb_coef) > 0) {
  all_coefs <- bind_rows(all_coefs, mb_coef)
  cat("\nModel B Coefficients:\n")
  mb_coef %>% select(term, estimate, std.error, odds_ratio, p.value, sig) %>% print(n = 20)
}

if (result_b$success) {
  saveRDS(result_b$model, "results_replication/model_b.rds")
  write_csv(mb_coef, "results_replication/model_b_coefficients.csv")
  cat("  Saved model B results.\n")
}

rm(model_b_data); gc()

# --- Empty Model (ICC) -------------------------------------------------------

cat("\n--- Empty Model (ICC Decomposition) ---\n")

empty_data <- model_data %>%
  filter(!is.na(prominence), !is.na(issue_area))
empty_data$org_id <- droplevels(empty_data$org_id)
empty_data$issue_area <- droplevels(empty_data$issue_area)

result_empty <- fit_glmer_safe(
  prominence ~ 1 + (1 | org_id) + (1 | issue_area),
  data = empty_data,
  model_name = "Empty Model"
)

model_results[["Empty"]] <- result_empty

if (result_empty$success && grepl("glmer", result_empty$method)) {
  tryCatch({
    vc <- as.data.frame(VarCorr(result_empty$model))
    total_var <- sum(vc$vcov) + (pi^2 / 3)
    cat("\nVariance Decomposition (ICC):\n")
    for (i in seq_len(nrow(vc))) {
      pct <- vc$vcov[i] / total_var * 100
      cat(sprintf("  %s: %.4f (%.1f%%)\n", vc$grp[i], vc$vcov[i], pct))
    }
    cat(sprintf("  Residual (logistic): %.4f (%.1f%%)\n",
                pi^2/3, (pi^2/3)/total_var * 100))
  }, error = function(e) {
    cat(sprintf("  Could not compute ICC: %s\n", e$message))
  })

  saveRDS(result_empty$model, "results_replication/model_empty.rds")
  cat("  Saved empty model.\n")
}

rm(empty_data); gc()

# --- Model Comparison --------------------------------------------------------

cat("\n--- Model Comparison ---\n")

comparison_rows <- list()
for (name in names(model_results)) {
  r <- model_results[[name]]
  if (r$success) {
    row <- tibble(
      Model = name,
      Method = r$method,
      AIC = tryCatch(AIC(r$model), error = function(e) NA_real_),
      BIC = tryCatch(BIC(r$model), error = function(e) NA_real_),
      N = tryCatch(nobs(r$model), error = function(e) NA_integer_),
      Converged = tryCatch({
        if (inherits(r$model, "glmerMod")) {
          is.null(r$model@optinfo$conv$lme4$messages)
        } else {
          r$model$converged
        }
      }, error = function(e) NA)
    )
    comparison_rows[[name]] <- row
  } else {
    comparison_rows[[name]] <- tibble(
      Model = name, Method = "FAILED", AIC = NA, BIC = NA, N = NA, Converged = FALSE
    )
  }
}

comparison <- bind_rows(comparison_rows)
cat("\n")
print(comparison)

# --- Save Combined Results ---------------------------------------------------

cat("\n--- Saving Combined Results ---\n")

if (nrow(all_coefs) > 0) {
  write_csv(all_coefs, "results_replication/glmm_replication_results.csv")
  cat("  Saved: results_replication/glmm_replication_results.csv\n")
}

write_csv(comparison, "results_replication/glmm_replication_comparison.csv")
cat("  Saved: results_replication/glmm_replication_comparison.csv\n")

# --- Forest Plot --------------------------------------------------------------

if (nrow(all_coefs) > 0) {
  cat("\n--- Generating Forest Plot ---\n")

  plot_data <- all_coefs %>%
    filter(term != "(Intercept)") %>%
    mutate(
      term = fct_reorder(term, estimate),
      model = factor(model, levels = c("C", "A", "B"))
    )

  if (nrow(plot_data) > 0) {
    p <- ggplot(plot_data, aes(x = estimate, y = term, color = model, shape = model)) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
      geom_point(position = position_dodge(width = 0.6), size = 2.5) +
      geom_errorbarh(
        aes(xmin = conf.low, xmax = conf.high),
        position = position_dodge(width = 0.6),
        height = 0.2
      ) +
      labs(
        title = "GLMM Replication: Coefficient Estimates",
        subtitle = "Random effects structure noted in Method column",
        x = "Log-Odds Coefficient (with 95% CI)",
        y = NULL,
        color = "Model",
        shape = "Model"
      ) +
      theme_minimal(base_size = 12) +
      theme(
        legend.position = "bottom",
        panel.grid.minor = element_blank()
      ) +
      scale_color_brewer(palette = "Set1")

    ggsave("results_replication/glmm_replication_forest.png", p,
           width = 10, height = 8, dpi = 150)
    cat("  Saved: results_replication/glmm_replication_forest.png\n")
  }
}

# --- Summary ------------------------------------------------------------------

cat("\n")
cat("========================================\n")
cat("GLMM REPLICATION RESULTS\n")
cat("========================================\n")

status_label <- function(result) {
  if (!result$success) return("FAILED")
  if (result$method == "glm_fixed") return("CONVERGED (fixed effects only)")
  if (result$method == "glmer_issue_only") return("CONVERGED (issue_area RE only)")
  if (inherits(result$model, "glmerMod")) {
    if (is.null(result$model@optinfo$conv$lme4$messages)) {
      return("CONVERGED")
    } else {
      return("CONVERGED (with warnings)")
    }
  }
  return("CONVERGED")
}

cat(sprintf("Model C (Group Characteristics): %s\n", status_label(model_results[["C"]])))
cat(sprintf("Model A (Policy Salience):       %s\n", status_label(model_results[["A"]])))
cat(sprintf("Model B (Group-Politician):      %s\n", status_label(model_results[["B"]])))
cat(sprintf("Empty Model (ICC):               %s\n", status_label(model_results[["Empty"]])))
cat("========================================\n")

converged_count <- sum(sapply(model_results[c("C", "A", "B")], function(r) r$success))
cat(sprintf("\nOverall: %d of 3 models fitted successfully.\n", converged_count))

if (nrow(all_coefs) > 0) {
  cat("\nKey Coefficients (all models):\n")
  all_coefs %>%
    filter(term != "(Intercept)") %>%
    arrange(model, desc(abs(estimate))) %>%
    select(model, method, term, estimate, odds_ratio, p.value, sig) %>%
    print(n = 30)
}

cat("\nResults saved to results_replication/\n")
cat("Done.\n")
