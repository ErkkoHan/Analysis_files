#!/usr/bin/env Rscript
# Fig. 1a study-site map. Requires establishment coordinates (lat, lng), which
# are not included in the public data; run export_analysis_sites.py first.

suppressPackageStartupMessages({
  library(sf)
  library(ggplot2)
  library(readr)
  library(rnaturalearth)
  library(ggspatial)
  library(grid)
})

# ---- Settings ---------------------------------------------------------------- #
WIDTH_MM  <- 57.8
HEIGHT_MM <- 104.0
DPI       <- 600
BASE_PT   <- 8
GRAT_STEP <- 0.5
MM        <- 1 / 25.4

script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) normalizePath(dirname(f)) else normalizePath(getwd())
}
HERE    <- script_dir()
OUT_DIR <- file.path(HERE, "output")

# ---- Sites in the analytical sample ---------------------------------------- #
raw <- readr::read_csv(file.path(HERE, "dataset", "full_data.csv"), show_col_types = FALSE)
names(raw) <- sub("^﻿", "", names(raw))
raw <- raw[is.finite(raw$distance_km) & raw$distance_km <= 100, ]
keep <- readr::read_csv(file.path(OUT_DIR, "analysis_sites.csv"), show_col_types = FALSE)
raw <- raw[raw$cid %in% keep$cid & is.finite(raw$lat) & is.finite(raw$lng), ]
cat(sprintf("%d establishments\n", length(unique(raw$cid))))
sites_sf <- st_as_sf(unique(raw[, c("lat", "lng")]), coords = c("lng", "lat"), crs = 4326)

# ---- Basemap: Taiwan main island and county boundaries ---------------------- #
crs_tw <- 3826
taiwan <- ne_countries(scale = "large", country = "Taiwan", returnclass = "sf")
taiwan_main <- st_cast(taiwan, "POLYGON", warn = FALSE)
taiwan_main <- taiwan_main[which.max(st_area(taiwan_main)), ]
taiwan_p <- st_transform(taiwan_main, crs_tw)
sites_p  <- st_transform(sites_sf, crs_tw)

states_p <- st_transform(ne_states(country = "Taiwan", returnclass = "sf"), crs_tw)
states_p <- states_p[lengths(st_intersects(states_p, taiwan_p)) > 0, ]
states_p <- suppressWarnings(st_intersection(st_make_valid(states_p), st_make_valid(taiwan_p)))
states_p <- st_collection_extract(states_p, "POLYGON")

bb  <- st_bbox(taiwan_p)
pad <- 0.02 * c(bb["xmax"] - bb["xmin"], bb["ymax"] - bb["ymin"])
xlim0 <- as.numeric(c(bb["xmin"] - pad[1], bb["xmax"] + pad[1]))
ylim0 <- as.numeric(c(bb["ymin"] - pad[2], bb["ymax"] + pad[2]))

# ---- Map builder ------------------------------------------------------------ #
inner_breaks <- function(lo, hi, step, frac_lo, frac_hi) {
  b <- seq(ceiling(lo / step) * step, floor(hi / step) * step, by = step)
  b[b >= lo + frac_lo * (hi - lo) & b <= hi - frac_hi * (hi - lo)]
}

build_map <- function(xlim, ylim) {
  ll <- st_bbox(st_transform(
    st_as_sfc(st_bbox(c(xmin = xlim[1], ymin = ylim[1], xmax = xlim[2], ymax = ylim[2]),
                      crs = crs_tw)), 4326))
  xb <- inner_breaks(ll[["xmin"]], ll[["xmax"]], GRAT_STEP, 0.03, 0.12)
  yb <- inner_breaks(ll[["ymin"]], ll[["ymax"]], GRAT_STEP, 0.02, 0.02)
  ggplot() +
    geom_sf(data = states_p, fill = "grey92", colour = "grey70", linewidth = 0.15) +
    geom_sf(data = taiwan_p, fill = NA, colour = "grey45", linewidth = 0.3) +
    geom_sf(data = sites_p, shape = 21, fill = "#c0392b", colour = "white",
            size = 1.2, stroke = 0.2, alpha = 0.9) +
    coord_sf(xlim = xlim, ylim = ylim, expand = FALSE, crs = crs_tw, datum = 4326) +
    scale_x_continuous(breaks = xb) +
    scale_y_continuous(breaks = yb) +
    annotation_scale(location = "bl", width_hint = 0.34, style = "ticks",
                     line_width = 0.5, height = unit(0.10, "cm"),
                     pad_x = unit(0.10, "cm"), pad_y = unit(0.10, "cm"),
                     text_cex = BASE_PT / 12) +
    annotation_north_arrow(location = "tr", which_north = "true",
                           height = unit(0.55, "cm"), width = unit(0.55, "cm"),
                           pad_x = unit(0.10, "cm"), pad_y = unit(0.10, "cm"),
                           style = north_arrow_orienteering(text_size = BASE_PT,
                                                            line_width = 0.5)) +
    labs(x = NULL, y = NULL) +
    theme_bw(base_size = BASE_PT) +
    theme(panel.grid = element_line(colour = "grey88", linewidth = 0.2),
          panel.border = element_rect(colour = "grey40", linewidth = 0.4),
          axis.text = element_text(size = BASE_PT, colour = "grey30"),
          axis.ticks = element_line(colour = "grey40", linewidth = 0.3),
          axis.ticks.length = unit(0.05, "cm"),
          plot.margin = margin(2, 2, 1, 1, "mm"))
}

# ---- Fit the map extent to the panel ---------------------------------------- #
w_in <- WIDTH_MM * MM
h_in <- HEIGHT_MM * MM
panel_size <- local({
  tmp <- tempfile(fileext = ".png")
  grDevices::png(tmp, width = w_in, height = h_in, units = "in", res = 150, bg = "white")
  on.exit({ grDevices::dev.off(); unlink(tmp) }, add = TRUE)
  gt <- ggplotGrob(build_map(xlim0, ylim0))
  fw <- sum(grid::convertWidth(gt$widths[grid::unitType(gt$widths) != "null"], "in",
                               valueOnly = TRUE))
  fh <- sum(grid::convertHeight(gt$heights[grid::unitType(gt$heights) != "null"], "in",
                                valueOnly = TRUE))
  c(w = w_in - fw, h = h_in - fh)
})
A  <- unname(panel_size["w"] / panel_size["h"])
w0 <- diff(xlim0)
h0 <- diff(ylim0)
if (w0 / h0 < A) { w1 <- h0 * A; h1 <- h0 } else { w1 <- w0; h1 <- w0 / A }
xlim <- mean(xlim0) + c(-1, 1) * w1 / 2
ylim <- mean(ylim0) + c(-1, 1) * h1 / 2
p <- build_map(xlim, ylim)

# ---- Export ------------------------------------------------------------------- #
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
ggsave(file.path(OUT_DIR, "Fig1a_study_sites.png"), p, width = WIDTH_MM, height = HEIGHT_MM,
       units = "mm", dpi = DPI, bg = "white")
ggsave(file.path(OUT_DIR, "Fig1a_study_sites.pdf"), p, width = WIDTH_MM, height = HEIGHT_MM,
       units = "mm", device = cairo_pdf, bg = "white")
