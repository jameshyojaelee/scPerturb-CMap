#!/usr/bin/env Rscript
# Fig 2: Recall@50 by cell line (R version)

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

output_png <- resolve_path(if (!is.null(opts[["output-png"]])) opts[["output-png"]] else "figs/fig2_recall_by_cell_line_r.png")
output_pdf <- resolve_path(if (!is.null(opts[["output-pdf"]])) opts[["output-pdf"]] else "figs/fig2_recall_by_cell_line_r.pdf")

df <- read.csv(input_path, stringsAsFactors = FALSE)
required_cols <- c("cell_line", "method", "recall_at_50", "seed")
missing <- setdiff(required_cols, names(df))
if (length(missing) > 0) stop("Input missing required columns: ", paste(missing, collapse = ", "))

df <- df %>% filter(!is.na(cell_line), !is.na(method), !is.na(recall_at_50))
if (!all(c("Baseline", "MetricBlend") %in% unique(df$method))) {
  stop("Input must contain both 'Baseline' and 'MetricBlend' methods.")
}

summary_tbl <- df %>%
  group_by(cell_line, method) %>%
  summarise(
    mean_recall = mean(recall_at_50),
    sd = sd(recall_at_50),
    n = dplyr::n(),
    .groups = "drop"
  ) %>%
  mutate(ci95 = ifelse(n > 1 & !is.na(sd), 1.96 * sd / sqrt(n), 0))

order_tbl <- summary_tbl %>%
  group_by(cell_line) %>%
  summarise(order_val = max(mean_recall), .groups = "drop")
summary_tbl <- summary_tbl %>%
  left_join(order_tbl, by = "cell_line") %>%
  mutate(cell_line = reorder(cell_line, order_val))

palette <- c("Baseline" = "#326799", "MetricBlend" = "#D66640")

p <- ggplot(summary_tbl, aes(x = cell_line, y = mean_recall, fill = method)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6, colour = "#20313C") +
  geom_errorbar(aes(ymin = mean_recall - ci95, ymax = mean_recall + ci95),
                position = position_dodge(width = 0.7), width = 0.18, linewidth = 0.6) +
  scale_fill_manual(values = palette) +
  labs(
    title = "Fig 2. Recall@50 by cell line",
    x = "Cell line",
    y = "Mean recall@50",
    fill = "Method"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 30, hjust = 1)
  ) +
  ylim(0, 1)

ensure_parent(output_png)
ensure_parent(output_pdf)
ggsave(output_png, plot = p, width = max(7, 0.55 * n_distinct(summary_tbl$cell_line) + 3), height = 5, dpi = 300)
ggsave(output_pdf, plot = p, width = max(7, 0.55 * n_distinct(summary_tbl$cell_line) + 3), height = 5)

cat("Summary table (mean ± 95% CI):\n")
print(summary_tbl %>%
        select(cell_line, method, mean_recall, ci95, n) %>%
        arrange(cell_line, method))
