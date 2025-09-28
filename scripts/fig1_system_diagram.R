#!/usr/bin/env Rscript
# Fig 1: scPerturb-CMap system diagram (R version)

suppressPackageStartupMessages(library(ggplot2))

parse_cli <- function(argv) {
  res <- list()
  i <- 1
  while (i <= length(argv)) {
    key <- argv[i]
    if (!startsWith(key, "--")) {
      stop("Unexpected argument: ", key)
    }
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

`%||%` <- function(x, y) if (is.null(x)) y else x

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
script_path <- if (length(file_arg) > 0) {
  normalizePath(sub("^--file=", "", file_arg))
} else {
  stop("Unable to determine script path (use Rscript to run).")
}
repo_root <- normalizePath(file.path(dirname(script_path), ".."))

argv <- commandArgs(trailingOnly = TRUE)
opts <- parse_cli(argv)

resolve_path <- function(path) {
  if (grepl("^(/|[A-Za-z]:)", path)) return(normalizePath(path, mustWork = FALSE))
  normalizePath(file.path(repo_root, path), mustWork = FALSE)
}

output_png <- resolve_path(if (!is.null(opts[["output-png"]])) opts[["output-png"]] else "figs/fig1_system_diagram_r.png")
output_pdf <- resolve_path(if (!is.null(opts[["output-pdf"]])) opts[["output-pdf"]] else "figs/fig1_system_diagram_r.pdf")

ensure_parent <- function(path) {
  dir <- dirname(path)
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE, showWarnings = FALSE)
}

boxes <- data.frame(
  name = c("Target JSON", "L1000 long", "Baseline", "DualEncoder", "Blended results"),
  label = c(
    "Target JSON\n(genes, weights, metadata)",
    "L1000 Level 5\nLandmarks + annotations",
    "Baseline scoring
Cosine inversion +\nGSEA (z)",
    "DualEncoder metric\nTrain inversion pairs\nNT-Xent / Triplet",
    "Blended score & ranking
Ranked compounds + MoA"
  ),
  xmin = c(0.2, 2.7, 5.35, 5.35, 8.55),
  xmax = c(2.1, 5.0, 6.65, 6.65, 10.25),
  ymin = c(3.35, 3.2, 5.95, 0.9, 3.25),
  ymax = c(4.95, 4.95, 6.95, 2.35, 4.95),
  fill = factor(c("Input", "Reference", "Analysis", "Analysis", "Output"),
                levels = c("Input", "Reference", "Analysis", "Output"))
)
boxes$xc <- (boxes$xmin + boxes$xmax) / 2
boxes$yc <- (boxes$ymin + boxes$ymax) / 2

arrows <- data.frame(
  x = c(boxes$xmax[1], boxes$xc[2], boxes$xc[2], boxes$xmax[3], boxes$xmax[4]),
  y = c(boxes$yc[1], boxes$ymax[2], boxes$ymin[2], boxes$yc[3], boxes$yc[4]),
  xend = c(boxes$xmin[2], boxes$xmin[3], boxes$xmin[4], boxes$xmin[5], boxes$xmin[5]),
  yend = c(boxes$yc[2], boxes$yc[3], boxes$yc[4], boxes$yc[5] + 0.9, boxes$yc[5] - 0.9),
  label = c("Align genes", "Baseline input", "Metric input", "Blend", "Blend"),
  lx = c(2.35, 4.05, 4.05, 7.4, 7.4),
  ly = c(boxes$yc[2] + 0.9, boxes$yc[3] + 0.85, boxes$yc[4] - 0.85, boxes$yc[5] + 1.35, boxes$yc[5] - 1.35)
)

palette <- c("Input" = "#7EA9E1", "Reference" = "#F4C27A", "Analysis" = "#90D1C2", "Output" = "#F5A6A6")

plot_df <- ggplot(boxes) +
  geom_rect(aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = fill),
            colour = "#2F3E46", linewidth = 0.7) +
  geom_curve(data = arrows,
             aes(x = x, y = y, xend = xend, yend = yend),
             curvature = 0.18, colour = "#2F3E46", linewidth = 0.8,
             arrow = arrow(length = unit(0.25, "cm"), type = "closed", angle = 25)) +
  geom_text(aes(x = xc, y = yc, label = label), lineheight = 0.96,
            family = "Helvetica", fontface = "bold", size = 3.5, colour = "#102542") +
  geom_label(data = arrows,
             aes(x = lx, y = ly, label = label),
             family = "Helvetica", size = 3.0, fill = "#F7F7F7", colour = "#102542",
             linewidth = 0, label.padding = unit(0.12, "lines")) +
  scale_fill_manual(values = palette, name = NULL) +
  coord_equal(xlim = c(-0.2, 10.6), ylim = c(0.2, 7.4)) +
  labs(title = "Fig 1. scPerturb-CMap pipeline") +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    legend.direction = "horizontal",
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5, colour = "#0C1821"),
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = margin(12, 12, 28, 12)
  ) +
  annotate("text", x = 4.8, y = 0.7,
           label = "scPerturb-CMap: targets align to L1000 signatures;\nBaseline and DualEncoder scores blend to rank compounds.",
           size = 3.4, colour = "#1B263B", family = "Helvetica", lineheight = 0.98)

ensure_parent(output_png)
ensure_parent(output_pdf)
ggsave(output_png, plot = plot_df, width = 10, height = 6.4, dpi = 300)
ggsave(output_pdf, plot = plot_df, width = 10, height = 6.4)
