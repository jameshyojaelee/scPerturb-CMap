#!/usr/bin/env Rscript
# Fig 4: Case study ranked compounds (R version)

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
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

top_k <- if (!is.null(opts[["top-k"]])) as.integer(opts[["top-k"]]) else 50L
label_top <- if (!is.null(opts[["label-top"]])) as.integer(opts[["label-top"]]) else 10L

output_png <- resolve_path(if (!is.null(opts[["output-png"]])) opts[["output-png"]] else "figs/fig4_case_study_rank_plot_r.png")
output_pdf <- resolve_path(if (!is.null(opts[["output-pdf"]])) opts[["output-pdf"]] else "figs/fig4_case_study_rank_plot_r.pdf")

df <- read.csv(input_path, stringsAsFactors = FALSE)
required_cols <- c("rank", "compound", "blended_score")
missing <- setdiff(required_cols, names(df))
if (length(missing) > 0) stop("Input missing required columns: ", paste(missing, collapse = ", "))

df <- df %>%
  mutate(
    rank = as.numeric(rank),
    blended_score = as.numeric(blended_score),
    compound = as.character(compound)
  ) %>%
  filter(!is.na(rank), !is.na(blended_score), !is.na(compound)) %>%
  arrange(rank) %>%
  head(max(1, top_k))

if (nrow(df) == 0) stop("No rows available after filtering.")

label_df <- head(df, max(0, label_top))

p <- ggplot(df, aes(x = rank, y = blended_score)) +
  geom_line(colour = "#2B6CB0", linewidth = 1) +
  geom_point(colour = "#1E4E8C", size = 2.8) +
  geom_text(data = label_df,
            aes(label = compound),
            vjust = -0.8, hjust = 0.5,
            size = 3, fontface = "bold", colour = "#1B263B") +
  labs(
    title = "Fig 4. Case study ranked compounds",
    x = "Rank",
    y = "Blended score"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5)
  )

ensure_parent(output_png)
ensure_parent(output_pdf)
ggsave(output_png, plot = p, width = 8, height = 5, dpi = 300)
ggsave(output_pdf, plot = p, width = 8, height = 5)

cat("Top 10 compounds:\n")
print(df %>% select(rank, compound, blended_score) %>% head(10))
