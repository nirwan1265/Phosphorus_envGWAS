# library
library(terra)
library(sf)
library(tmap)
library(rnaturalearth)
library(rnaturalearthdata)
library(classInt)
library(viridisLite)

# Load the raster
r <- rast("/Users/nirwantandukar/Documents/Research/data/P_prediction/predictors/nitrogen_0-5cm_mean_1000.tif")

#  Central America boundary
ctrs <- c("Mexico","Guatemala","Belize","Honduras","El Salvador","Nicaragua","Costa Rica","Panama")
borders <- rnaturalearth::ne_countries(country = ctrs, scale = "medium", returnclass = "sf")
borders <- st_transform(borders, crs(r))
ca_poly <- st_union(borders)

#  crop & mask to CA
r_ca <- mask(crop(r, vect(ca_poly)), vect(ca_poly))

# Units label (SoilGrids N is typically g kg^-1; change if yours differs)
unit_lab <- "g kg\u207B\u00B9"

# Breaks & palette
vals <- values(r_ca, na.rm = TRUE)
brks <- classInt::classIntervals(vals, n = 8, style = "quantile")$brks  # 7 classes (8 breaks)

# Plot
tmap_mode("plot")
map <- tm_shape(r_ca) +
  tm_raster(style = "fixed",
            breaks = brks,
            palette = viridis(length(brks) - 1),
            title = paste0("Soil nitrogen (0–5 cm, ", unit_lab, ")")) +
  tm_shape(borders) + tm_borders(col = "grey20", lwd = 0.6) +
  tm_layout(#title = "Central America",
            title.position = c("center","top"),
            legend.outside = TRUE,
            legend.outside.position = "right",
            legend.frame = TRUE,
            frame = TRUE,
            bg.color = "white") +
  tm_scale_bar(position = c("left","bottom")) +
  tm_compass(type = "8star", position = c("right","top"), size = 2) +
  tm_grid(alpha = 0.2, col = "grey70", labels.size = 0)

map

# Save
tmap_save(map, "central_america_nitrogen_0_5cm.png",
          dpi = 300, width = 9, height = 6, units = "in")


