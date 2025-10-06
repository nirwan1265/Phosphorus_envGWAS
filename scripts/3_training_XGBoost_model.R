#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
################################################################################
#  Train the model
################################################################################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


### Load the training data with predictors
data <- readRDS("output/Training_data_OlsenP_with_predictors_with_conversion.RDS")

# old
# data <- readRDS("output/Training_data_OlsenP_with_predictors_old.RDS")
# colnames(data)
#remove Long lat
data <- data[,-c(2,3)]
colnames(data)


library(dplyr)
library(xgboost)
library(rBayesianOptimization)

set.seed(20)

# 0) target transform
data$OlsenP <- log10(data$OlsenP)

data <- data %>%
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.numeric)


# 1) split 80/20
n   <- nrow(data)
idx <- sample.int(n, size = floor(0.8 * n))
train <- data[idx, , drop = FALSE]
test  <- data[-idx, , drop = FALSE]

# 2) X/y
y_train <- train$OlsenP
X_train <- train %>% dplyr::select(-OlsenP)
y_test  <- test$OlsenP
X_test  <- test  %>% dplyr::select(-OlsenP)



# Convert data to matrix, as xgboost doesn't accept data frames
set.seed(1)
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train, missing = NA)


################# Optimizing parameters
# Define the objective function to be optimized
# Initialize a global variable to store problematic parameters
problematic_params <- list()

# Modify the objective function to skip errors and log problematic parameter sets
objective_function <- function(max_depth, min_child_weight, subsample, colsample_bytree, gamma, alpha, lambda) {
  params <- list(
    booster = "gbtree",
    eta = 0.1,
    max_depth = as.integer(max_depth),
    min_child_weight = min_child_weight,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    gamma = gamma,
    eval_metric = "rmse",
    objective = "reg:squarederror",
    lambda = lambda,
    alpha = alpha,
    nthread = 8  # Adjust based on your setup
  )
  
  # Try-catch block to handle errors during model training
  result <- tryCatch({
    cv_results <- xgb.cv(
      params = params,
      data = dtrain,  # Ensure dtrain is correctly defined
      nrounds = 100,
      nfold = 5,
      showsd = TRUE,
      stratified = FALSE,
      print_every_n = 10,
      early_stopping_rounds = 10,
      maximize = FALSE
    )
    score <- -min(cv_results$evaluation_log$test_rmse_mean)
    if(is.finite(score)) {
      return(list(Score = score, Pred = min(cv_results$evaluation_log$test_rmse_mean)))
    } else {
      stop("Non-finite value encountered.")  # Trigger error handling
    }
  }, error = function(e) {
    # Log problematic parameters and the error
    cat("Error with parameters:", toString(params), "Error message:", e$message, "\n")
    # Append the problematic parameters and error message to the global list
    problematic_params <<- c(problematic_params, list(list(params = params, error = e$message)))
    return(list(Score = -Inf, Pred = Inf))  # Return a high penalty for error cases
  })
  
  return(result)
}


# Define the bounds of the hyperparameters
bounds <- list(
  max_depth        = c(6L, 10L),
  min_child_weight = c(4, 9),
  subsample        = c(0.6, 0.9),
  colsample_bytree = c(0.6, 0.9),
  gamma            = c(0.2, 1.0),
  alpha            = c(1, 15),
  lambda           = c(1, 15)
)

bounds <- list(
  max_depth        = c(4L, 10L),
  min_child_weight = c(1, 15),
  subsample        = c(0.6, 0.9),
  colsample_bytree = c(0.6, 0.9),
  gamma            = c(0.2, 1.0),
  alpha            = c(1, 15),
  lambda           = c(1, 15)
)


# Run Bayesian Optimization
bayes_opt_result <- BayesianOptimization(
  FUN = objective_function,
  bounds = bounds,
  init_points = 5,  # Number of randomly chosen points to sample the target function before fitting the Gaussian process
  n_iter = 100,      # Number of iterations to perform
  acq = "ucb",       # Acquisition function type: expected improvement, ei to ucb
  kappa = 2.5,           # Higher kappa for more exploration
  verbose = TRUE
)

# Print the best parameters and the corresponding RMSE
print(bayes_opt_result$Best_Par)
#Best Parameters Found: 
#  Round = 39	max_depth = 10.0000	min_child_weight = 5.558417	subsample = 0.8913824	colsample_bytree = 0.9000	gamma = 2.220446e-16	alpha = 2.220446e-16	lambda = 6.489022	Value = -0.3399588 

#old
#max_depth min_child_weight        subsample colsample_bytree            gamma            alpha           lambda 
#9.0000000        4.6471442        0.7036718        0.6125160        0.4295447       12.6034985       10.4192541 

# normalized
#Best Parameters Found: 
#  Round = 55	max_depth = 10.0000	min_child_weight = 1.0000	subsample = 0.9000	colsample_bytree = 0.9000	gamma = 1.0000	alpha = 15.0000	lambda = 14.99292	Value = -21.30874 

# Parameters
params <- list(
  booster = "gbtree",
  eta = 0.1,
  max_depth = 10.0000000,
  min_child_weight = 1,
  subsample = 0.9,
  colsample_bytree = 0.9,
  eval_metric = "rmse",
  objective = "reg:squarederror",
  lambda = 14.99292,  # L2 regularization
  alpha = 15 ,    # L1 regularization
  gamma = 1,        # Minimum loss reduction required to make a further partition
  nthread = 8,
  num_parallel_tree = 1  # Use more than 1 for boosted random forests
)


# Find the best nrounds:
### FOR CV and nrounds:
# Perform 10-fold cross-validation
cv.nfold <- 10
cv.nrounds <- 1000
set.seed(123) # For reproducibility
cv_results <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = cv.nrounds,
  nfold = cv.nfold,
  showsd = TRUE,
  stratified = TRUE,
  print_every_n = 10,
  early_stopping_rounds = 10,
  maximize = FALSE
)

# Review the cross-validation results
print(cv_results)
#saveRDS(cv_results,"XGBoost_CV_results_allpredictors.RDS")

#Best iteration:
#Best iteration:
#  iter train_rmse_mean train_rmse_std test_rmse_mean test_rmse_std
#<num>           <num>          <num>          <num>         <num>
#  264       0.1110501    0.001527127      0.3369714   0.003528262

# Old:
#Best iteration:
#Stopping. Best iteration:
#  [61]	train-rmse:378.225980+11.068970	test-rmse:617.779042+149.458296

# Conversion
#Stopping. Best iteration:
#  [79]	train-rmse:14.092909+0.288934	test-rmse:21.355623+3.148758

## Running the model:
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 79)
saveRDS(xgb_model,"XGBoost_model_allpredictors_conversion.RDS")

## importance matrix
importance_matrix <- xgb.importance(feature_names = colnames(X_train), model = xgb_model)
print(importance_matrix)


# Make a test DMatrix
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test, missing = NA)

# ---- Predict ----
pred_test_log <- predict(xgb_model, dtest)

# ---- Metrics on log10 scale (this is the optimization scale) ----
rmse_log <- sqrt(mean((pred_test_log - y_test)^2, na.rm = TRUE))
mae_log  <- mean(abs(pred_test_log - y_test), na.rm = TRUE)
r2_log   <- 1 - sum((pred_test_log - y_test)^2, na.rm = TRUE) /
  sum((y_test - mean(y_test, na.rm = TRUE))^2, na.rm = TRUE)
rsq_log  <- cor(pred_test_log, y_test, use = "complete.obs")^2  # squared Pearson r

# ---- Back-transform to original Olsen-P scale ----
pred_test_lin <- 10^pred_test_log
y_test_lin    <- 10^y_test

rmse_lin <- sqrt(mean((pred_test_lin - y_test_lin)^2, na.rm = TRUE))
mae_lin  <- mean(abs(pred_test_lin - y_test_lin), na.rm = TRUE)
r2_lin   <- 1 - sum((pred_test_lin - y_test_lin)^2, na.rm = TRUE) /
  sum((y_test_lin - mean(y_test_lin, na.rm = TRUE))^2, na.rm = TRUE)
rsq_lin  <- cor(pred_test_lin, y_test_lin, use = "complete.obs")^2

# ---- Optional: bias-correct the back-transform (Duan smearing on base-10 logs) ----
pred_train_log <- predict(xgb_model, dtrain)
smear_factor   <- mean(10^(y_train - pred_train_log), na.rm = TRUE)  # ~exp(mean residual on base-10 scale)
pred_test_lin_bc <- pred_test_lin * smear_factor

rmse_lin_bc <- sqrt(mean((pred_test_lin_bc - y_test_lin)^2, na.rm = TRUE))
mae_lin_bc  <- mean(abs(pred_test_lin_bc - y_test_lin), na.rm = TRUE)
r2_lin_bc   <- 1 - sum((pred_test_lin_bc - y_test_lin)^2, na.rm = TRUE) /
  sum((y_test_lin - mean(y_test_lin, na.rm = TRUE))^2, na.rm = TRUE)
rsq_lin_bc  <- cor(pred_test_lin_bc, y_test_lin, use = "complete.obs")^2

# ---- Print a compact report ----
metrics <- list(
  log10_scale = c(RMSE = rmse_log, MAE = mae_log, R2 = r2_log, Rsq = rsq_log),
  linear_scale = c(RMSE = rmse_lin, MAE = mae_lin, R2 = r2_lin, Rsq = rsq_lin),
  linear_scale_bias_corrected = c(RMSE = rmse_lin_bc, MAE = mae_lin_bc, R2 = r2_lin_bc, Rsq = rsq_lin_bc)
)
print(metrics)


