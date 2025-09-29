#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
################################################################################
#  Load Data
################################################################################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

### TRAINING DATA

## Load the Olsen P data from McDowells and Mexico white paper data
data_olsenP <- read.csv("/Users/nirwantandukar/Documents/Research/data/P_prediction/phenotypes/Final_filtered_data_McDowell_Mexico.csv") %>% dplyr::select(OlsenP,Lat,Long)
data_olsenP <- data_olsenP[complete.cases(data_olsenP), ]
colnames(data_olsenP)


### PREDICTORS

### Load the predictors
dir_raster <- "/Users/nirwantandukar/Documents/Research/data/P_prediction/predictors"

# raster files:
raster_files <- list.files(path = dir_raster, pattern = "\\.tif$", full.names = TRUE)

# Function - extracting values from a raster file for all coordinates 
extract_raster_values <- function(raster_file, df) {
  r <- raster::brick(raster_file)
  
  coords <- df[, c("Long", "Lat")]
  coords_sp <- sp::SpatialPoints(coords, proj4string = sp::CRS("+proj=longlat +datum=WGS84"))
  coords_transformed <- sp::spTransform(coords_sp, crs(r))
  
  values <- raster::extract(x = r, y = coords_transformed)
  return(as.vector(values))
}


# Getting the values for the tif files
for (raster_file in raster_files) {
  # Get the variable name from the file name
  var_name <- gsub("\\.tif$", "", basename(raster_file))
  
  # Extract values for all coordinates in the pheno sorghum_35below frame
  extracted_values <- extract_raster_values(raster_file, data_olsenP)
  
  # Add the extracted values as a new column in the pheno sorghum_35below frame
  data_olsenP[[var_name]] <- extracted_values
}


### SAVE THE TRAINING DATA WITH PREDICTORS

saveRDS(data_olsenP, "output/Training_data_OlsenP_with_predictors.RDS")




















######### PREDICTION
### Load sorghum georef data (all)
data <- read.csv("users/Downloads/phosphorus_prediction/raw_data/Final_filtered_data.csv") %>% dplyr::select(OlsenP,Lat,Long,Continent)
data <- data[complete.cases(data), ]

dir_sorghum <- "data/"
sorghum <- read.csv(paste0(dir_sorghum,"taxa_geoloc_pheno.csv")) %>% 
  dplyr::filter(
    sp == "Sorghum bicolor",
    #GEO3 %in% c("Central Africa", "Eastern Africa", "North Africa", "Southern Africa", "Western Africa")
  ) %>%
  dplyr::select(2,6,7)

sorghum$GEO3major <- "NA"
sorghum$p_avg <- 0
rownames(sorghum) <- sorghum$Taxa
sorghum <- sorghum[,-1]
str(sorghum)
