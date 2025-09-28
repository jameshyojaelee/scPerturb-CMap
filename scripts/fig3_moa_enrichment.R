#!/usr/bin/env Rscript
# Fig 3: MoA enrichment visualisation (R version)

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

parse_cli <- function(argv) {
  res <- list()
  i <- 1
  while (i <= length(argv)) {
    key <- argv[i]
    if (!startsWith(key, "--")) stop("Unexpected argument: ", key)
    if (i == length(argv) || startsWith(argv[i + 1], "--")) {
      res[[substr(key, 3, nchar(key))]] <- TRUE
      i <- i + 1
    } else {
      res[[substr(key, 3, nchar(key))]] <- argv[i + 1]
      i <- i + 2
    }
  }
  res
}

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
script_path <- if (length(file_arg) > 0) normalizePath(sub("^--file=", "", file_arg)) else stop("Use Rscript to run.")
repo_root <- normalizePath(file.path(dirname(script_path), ".."))

argv <- commandArgs(trailingOnly = TRUE)
opts <- parse_cli(argv)

resolve_path <- function(path) {
  if (is.null(path)) return(NULL)
  if (grepl("^(/|[A-Za-z]:)", path)) return(normalizePath(path, mustWork = FALSE))
  normalizePath(file.path(repo_root, path), mustWork = FALSE)
}

ensure_parent <- function(path) {
  dir <- dirname(path)
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE, showWarnings = FALSE)
}

input_path <- resolve_path(opts[["input"]])
if (is.null(input_path)) stop("--input is required")

mode <- opts[["mode"]]
if (isTRUE(mode)) mode <- "bar"
mode <- tolower(ifelse(is.null(mode), "bar", mode))
if (!mode %in% c("bar", "volcano")) stop("--mode must be 'bar' or 'volcano'")

top_n <- if (!is.null(opts[["top-k"]])) as.integer(opts[["top-k"]]) else 20L
alpha <- if (!is.null(opts[["alpha"]])) as.numeric(opts[["alpha"]]) else 0.05

output_png <- resolve_path(if (!is.null(opts[["output-png"]])) opts[["output-png"]] else "figs/fig3_moa_enrichment_r.png")
output_pdf <- resolve_path(if (!is.null(opts[["output-pdf"]])) opts[["output-pdf"]] else "figs/fig3_moa_enrichment_r.pdf")

df <- read.csv(input_path, stringsAsFactors = FALSE)
required_cols <- c("moa", "log2_enrichment", "p_value")
missing <- setdiff(required_cols, names(df))
if (length(missing) > 0) stop("Input missing required columns: ", paste(missing, collapse = ", "))

df <- df %>%
  mutate(
    moa = as.character(moa),
    log2_enrichment = as.numeric(log2_enrichment),
    p_value = as.numeric(p_value),
    fdr = if ("fdr" %in% names(.)) as.numeric(fdr) else NA_real_
  ) %>%
  filter(!is.na(moa), !is.na(log2_enrichment), !is.na(p_value))

if (nrow(df) == 0) stop("No valid rows after cleaning.")

df <- df %>%
  mutate(significant = ifelse(!is.na(fdr), fdr <= alpha, p_value <= alpha))

ensure_parent(output_png)
ensure_parent(output_pdf)

if (mode == "bar") {
  plot_df <- df %>%
    mutate(abs_log2 = abs(log2_enrichment)) %>%
    arrange(desc(abs_log2)) %>%
    head(max(1, top_n)) %>%
    mutate(moa = reorder(moa, log2_enrichment))

  palette <- c("TRUE" = "#D94F58", "FALSE" = "#49759C")

  p <- ggplot(plot_df, aes(x = moa, y = log2_enrichment, fill = as.factor(significant))) +
    geom_col(width = 0.7, colour = "#1F2E3D") +
    coord_flip() +
    scale_fill_manual(values = palette, name = "Significant") +
    labs(
      title = sprintf("Fig 3. MoA enrichment (top %d)", nrow(plot_df)),
      x = "MoA",
      y = "log2 enrichment"
    ) +
    geom_hline(yintercept = 0, colour = "#233142", linetype = "dashed", linewidth = 0.6) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "top")

  ggsave(output_png, plot = p, width = 8, height = max(4, 0.25 * nrow(plot_df) + 2), dpi = 300)
  ggsave(output_pdf, plot = p, width = 8, height = max(4, 0.25 * nrow(plot_df) + 2))
} else {
  plot_df <- df %>% filter(p_value > 0) %>% mutate(neg_log10_p = -log10(p_value))
  if (nrow(plot_df) == 0) stop("No positive p-values for volcano mode")

  palette <- c("TRUE" = "#D94F58", "FALSE" = "#49759C")

  p <- ggplot(plot_df, aes(x = log2_enrichment, y = neg_log10_p, colour = as.factor(significant))) +
    geom_point(size = 2.4, alpha = 0.85) +
    scale_colour_manual(values = palette, name = "Significant") +
    geom_vline(xintercept = 0, colour = "#233142", linetype = "dashed", linewidth = 0.6) +
    geom_hline(yintercept = -log10(alpha), colour = "#E27D60", linetype = "dotted", linewidth = 0.7) +
    labs(
      title = "Fig 3. MoA enrichment volcano",
      x = "log2 enrichment",
      y = "-log10 p value"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "top")

  ggsave(output_png, plot = p, width = 7, height = 5.5, dpi = 300)
  ggsave(output_pdf, plot = p, width = 7, height = 5.5)
}

sig_count <- sum(df$significant, na.rm = TRUE)
cat(sprintf("Significant MoAs: %d\n", sig_count))
