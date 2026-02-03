# run_analysis.R
# Script to render the R Markdown multilevel analysis report

# Set working directory to R_analysis folder
if (!grepl("R_analysis$", getwd())) {
  if (file.exists("R_analysis")) {
    setwd("R_analysis")
  }
}

# Create output directories if they don't exist
output_dirs <- c(
  "../outputs/figures",
  "../outputs/tables",
  "../outputs/models",
  "../outputs/reports"
)

for (dir in output_dirs) {
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
    cat("Created directory:", dir, "\n")
  }
}

# Check for required packages
required_packages <- c(
  "tidyverse", "lme4", "broom.mixed", "sjPlot",
  "performance", "ggeffects", "knitr", "kableExtra", "rmarkdown"
)

missing <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]

if (length(missing) > 0) {
  cat("Missing packages:", paste(missing, collapse = ", "), "\n")
  cat("Installing missing packages...\n")
  install.packages(missing)
}

# Render the R Markdown report
cat("\n========================================\n")
cat("Rendering Multilevel Analysis Report\n")
cat("========================================\n\n")

tryCatch({
  rmarkdown::render(
    "Multilevel_Analysis.Rmd",
    output_dir = "../outputs/reports",
    output_format = "html_document"
  )
  cat("\n[SUCCESS] Report generated: outputs/reports/Multilevel_Analysis.html\n")
}, error = function(e) {
  cat("\n[ERROR] Failed to render report:\n")
  cat(e$message, "\n")
})
