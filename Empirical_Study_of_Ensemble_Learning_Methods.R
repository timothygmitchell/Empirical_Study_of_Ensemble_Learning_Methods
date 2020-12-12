### Getting Started
rm(list = ls())

# models
library(ranger)       # random forest, extremely randomized trees
library(xgboost)      # xgboost
library(MASS)         # linear discriminant analysis, quadratic discriminant analysis
library(mda)          # flexible discriminant analysis, mixture discriminant analysis
library(neuralnet)    # artificial neural networks
library(class)        # k-nearest neighbors
library(glmnet)       # elastic net
library(kernlab)      # gaussian processes
library(e1071)        # support vector machines, naive bayes
library(partykit)     # conditional inference tree

# misc
library(caret)        # confusionMatrix()
library(parallel)     # parallel computing
library(foreach)      # parallel computing
library(doParallel)   # parallel computing
library(tictoc)       # time processes
library(beepr)        # sound an alert when a process is finished

# helper functions
cv_parallel <- function(n, expr, ...) {
  
  # function to perform parallel repeated k-fold cross-validation
  # accepts as input a function kfcv() (defined for each base learner)
  rowMeans(simplify2array(mclapply(
    integer(n), eval.parent(substitute(function(...) expr)), ...)))
}

# Cohen's Kappa: a loss function appropriate for imbalanced classes
Kappa <- function(pred, actual) {confusionMatrix(pred, actual)[[3]][2]}

# convert any binary outcome to a factor with levels 0 and 1
binary <- function(vec) {factor(ifelse(vec == unique(vec)[1], 1, 0))}

# center and scale numeric variables (and leave factors alone)
standardize <- function(df) {rapply(df, scale, c("numeric", "integer"), how = "replace")}

# convert to a design matrix with one-hot-encoding
ohe <- function(df) {model.matrix(Y ~ 0 + ., df)}

# one-hot-encoding (dataframe output)
ohe.data.frame <- function(df) {data.frame(Y = df$Y, model.matrix(Y ~ 0 + ., df))}

phon <- "https://raw.githubusercontent.com/jbrownlee/Datasets/master/phoneme.csv"
phon <- read.csv(phon)[ , c(6, 1:5)]
colnames(phon) <- c("Y", paste0("X", 1:5))
phon$Y <- binary(phon$Y)

spam <- "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
spam <- read.csv(spam)[ , c(58, 1:57)]
colnames(spam) <- c("Y", paste0("X", 1:57))
spam$Y <- binary(spam$Y)

wdbc <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
wdbc <- read.csv(wdbc)[ , -1]
colnames(wdbc) <- c("Y", paste0("X", 1:30))
wdbc$Y <- binary(wdbc$Y)

adult <- "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
adult <- read.csv(adult)[ , c(15, 1:14)]
colnames(adult) <- c("Y", paste0("X", 1:14))
adult$Y <- binary(adult$Y)

park <- "http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
park <- read.csv(park)[ , c(18, 2:17, 19:24)]
colnames(park) <- c("Y", paste0("X", 1:22))
park$Y <- binary(park$Y)

# For faster computations, make k, c, and g smaller

k <- 5 # number of folds in cross-validation
c <- 4 # number of times to repeat cross-validation
g <- 50 # max rows to sample in random grid search

df <- wdbc # current data set

#########################################################################################

### Parameter Tuning: XGBoost

xgb.DMatrix <- xgb.DMatrix(sparse.model.matrix(Y ~ 0 + ., data = df), 
                           label = as.numeric(df$Y) - 1)

searchGrid <- expand.grid(subsample = c(0.40, 0.55, 0.70, 0.85, 1.0),
                          colsample_bytree = c(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                          colsample_bynode = c(0.5, 0.75, 1),
                          max_depth = c(4, 6, 8, 10, 12, 14, 16),
                          eta = c(0.001, 0.01, .1, 0.3))

searchGrid <- head(searchGrid[sample(1:nrow(searchGrid), nrow(searchGrid)), ], g*3)

cv_error <- foreach (i = 1:c, .combine = 'list', .multicombine = TRUE) %do% {
  
  tictoc::tic()
  
  cv_error <- t(apply(searchGrid, 1, function(params) {
    
    cv_log <- xgb.cv(data = xgb.DMatrix,                               # training sample
                     nround = 5000,                                    # maximum number of trees
                     early_stopping_rounds = 20,                       # stopping threshold if no improvement
                     objective = "binary:logistic",                    # objective function
                     eval_metric = "error",                            # evaluation metric
                     maximize = FALSE,                                 # want to MINIMIZE error
                     max.depth = params[["max_depth"]],                # tree depth
                     eta = params[["eta"]],                            # learning rate
                     gamma = 0,                                        # minimum loss reduction
                     subsample = params[["subsample"]],                # sample fraction of original data
                     colsample_bytree = params[["colsample_bytree"]],  # how many features sampled, each tree
                     colsample_bynode = params[["colsample_bynode"]],  # how many features sampled, each node
                     verbose = FALSE,
                     showsd = FALSE,
                     nfold = k,                                        # number of cv folds
                     stratified = TRUE)                                # stratify folds to balance classes
    
    best_error <- min(cv_log$evaluation_log[ , test_error_mean])
    best_rounds <- match(best_error, cv_log$evaluation_log[ , test_error_mean])
    
    return(c("error" = best_error, "trees" = best_rounds, params))
    
  })); tictoc::toc()
  
  return(cv_error)
}

cv_error <- apply(simplify2array(cv_error), 1:2, mean)    # mean error across all cycles

head(cv_error[order(cv_error[ , 1]), ])                   # minimum error

xgb.opt <- cv_error[order(cv_error[ , 1])[1], ]           # optimal hyperparameters

#########################################################################################

### Parameter Tuning: Extremely Randomized Trees

searchGrid <- expand.grid("mtry" = unique(floor(seq(5, ncol(df) - 1, length.out = 15))), 
                          "max.depth" = c(4, seq(5, 50, by = 5)))

searchGrid <- head(searchGrid[sample(1:nrow(searchGrid), nrow(searchGrid)), ], g)

# k-fold cross-validation
kfcv <- function(i = k) {
  
  fold <- sample(cut(1:nrow(df), breaks = i, labels = FALSE))
  
  cv_error <- foreach (i = 1:k, .combine = 'cbind') %do% {
    
    train <- df[fold != i, ]
    test <- df[fold == i, -1]
    
    mapply(function(x, y) {
      
      model <- ranger(Y ~ . , 
                      splitrule = "extratrees", 
                      replace = F, 
                      sample.fraction = 1,
                      data = train, 
                      num.trees = 300, 
                      mtry = x, 
                      max.depth = y)
      
      Kappa(predict(model, test)$predictions, df$Y[fold == i])
      
    }, x = searchGrid$mtry, y = searchGrid$max.depth)
    
  }
  
  return(rowMeans(cv_error))
  
}

tic(); cv_error <- cbind(searchGrid, kappa = cv_parallel(max(c, 3), kfcv())); toc()

head(cv_error[rev(order(cv_error$kappa)), ])            # minimum error

xt.opt <- cv_error[which.max(cv_error$kappa), ]         # optimal hyperparameters

#########################################################################################

### Parameter Tuning: Random Forest

searchGrid <- expand.grid("mtry" = unique(floor(seq(5, ncol(df) - 1, length.out = 15))), 
                          "max.depth" = c(4, seq(5, 50, by = 5)))

searchGrid <- head(searchGrid[sample(1:nrow(searchGrid), nrow(searchGrid)), ], g)

# k-fold cross-validation
kfcv <- function(i = k) {
  
  fold <- sample(cut(1:nrow(df), breaks = i, labels = FALSE))
  
  cv_error <- foreach (i = 1:k, .combine = 'cbind') %do% {
    
    train <- df[fold != i, ]
    test <- df[fold == i, -1]
    
    mapply(function(x, y) {
      
      model <- ranger(Y ~ . , 
                      data = train, 
                      num.trees = 300, 
                      mtry = x, 
                      max.depth = y)
      
      Kappa(predict(model, test)$predictions, df$Y[fold == i])
      
    }, x = searchGrid$mtry, y = searchGrid$max.depth)
    
  }
  
  return(rowMeans(cv_error))
  
}

tic(); cv_error <- cbind(searchGrid, kappa = cv_parallel(max(c, 3), kfcv())); toc()

head(cv_error[rev(order(cv_error$kappa)), ])            # minimum error

rf.opt <- cv_error[which.max(cv_error$kappa), ]         # optimal hyperparameters

#########################################################################################

### Parameter Tuning: Elastic Net

alpha <-seq(0, 1, by = 0.025) # elastic net mixing parameter

# function to find best shrinkage parameter (learning rate) for each value of alpha
best_lambda <- function(alpha) {
  
  # glmnet standardizes variables internally
  cv.glmnet(ohe(df), 
            df$Y,
            type.measure = "deviance",
            alpha = alpha, 
            nfolds = k,
            family = "binomial")$lambda.min}

tictoc::tic()
lambda <- mclapply(alpha, function(x) {mean(replicate(c, best_lambda(x)))})
tictoc::toc()

searchGrid <- data.frame(alpha = alpha, lambda = unlist(lambda))

kfcv <- function(i = k) {
  
  fold <- sample(cut(1:nrow(df), breaks = i, labels = FALSE))
  
  cv_error <- foreach (i = 1:k, .combine = 'cbind') %do% {
    
    train <- ohe(df)[fold != i, ]
    test <- ohe(df)[fold == i, ]
    
    mapply(function(x, y) {
      
      model <- glmnet(train, df$Y[fold != i], 
                      alpha = x, 
                      lambda = y, 
                      family = "binomial")
      
      Kappa(factor(predict(model, test, type = "class")), 
            df$Y[fold == i])
      
    }, x = searchGrid$alpha, y = searchGrid$lambda)
    
  }
  
  return(rowMeans(cv_error))
  
}

tic(); cv_error <- cbind(searchGrid, kappa = cv_parallel(c, kfcv())); toc()

head(cv_error[rev(order(cv_error$kappa)), ])        # minimum error

en.opt <- cv_error[which.max(cv_error$kappa), ]     # optimal hyperparameters

#########################################################################################

### Parameter Tuning: k-Nearest Neighbors

neighbors <- min(g, 30, floor(((k-1)/k)*nrow(df)) - 1)

neighbors <- sort(sample(neighbors)[1:g])

# k-fold cross-validation
kfcv <- function(i = k) {
  
  fold <- sample(cut(1:nrow(df), breaks = i, labels = FALSE))
  
  cv_error <- foreach (i = 1:k, .combine = 'cbind') %do% {
    
    train <- ohe(standardize(df))[fold != i, ]
    test <- ohe(standardize(df))[fold == i, ]
    
    unlist(mclapply(neighbors, function(x) {Kappa(knn(train, test, 
                                                      cl = df$Y[fold != i], k = x),
                                                  df$Y[fold == i])}, 
                    mc.cores = detectCores() - 1))
    
  }
  
  return(rowMeans(cv_error))
  
}

tic()
cv_error <- data.frame(neighbors, kappa = rowMeans(replicate(c, kfcv()))) 
toc()

head(cv_error[rev(order(cv_error$kappa)), ])       # minimum error

knn.opt <- cv_error[which.max(cv_error$kappa), ]   # optimal hyperparameters

#########################################################################################

# Parameter Tuning: Single Hidden Layer Neural Network

searchGrid <- expand.grid(hidden = floor(seq(0.25*(ncol(ohe(df))), 
                                             1.50*(ncol(ohe(df))), 
                                             by = 1)), 
                          algorithm = c("rprop+", "rprop-"))

searchGrid <- head(searchGrid[sample(1:nrow(searchGrid), nrow(searchGrid)), ], g)

kfcv <- function(i = k) {
  
  fold <- sample(cut(1:nrow(df), breaks = i, labels = FALSE))
  
  cv_error <- foreach (i = 1:k, .combine = 'cbind') %do% {
    
    train <- ohe.data.frame(standardize(df))[fold != i, ]
    
    test <- ohe.data.frame(standardize(df))[fold == i, -1]
    
    mapply(function(x, y) {
      
      nn <- neuralnet(Y ~ . , 
                      data = train, 
                      hidden = x, 
                      algorithm = y, 
                      rep = 1,
                      stepmax = 1e5,
                      threshold = 0.3,
                      linear.output = FALSE)
      
      Kappa(factor(round(predict(nn, test)[ , 2])), df$Y[fold == i])
      
    }, x = searchGrid$hidden, y = searchGrid$algorithm)
    
  }
  
  return(rowMeans(cv_error))
  
}

tic(); cv_error <- cbind(searchGrid, kappa = cv_parallel(c, kfcv())); toc()

head(cv_error[rev(order(cv_error$kappa)), ])        # minimum error

nn.opt <- cv_error[which.max(cv_error$kappa), ]     # optimal hyperparameters

#########################################################################################

### Parameter Tuning: Conditional Inference Tree

searchGrid <- expand.grid(alpha = c(0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25),
                          maxdepth = c(1:5, Inf))

searchGrid <- head(searchGrid[sample(1:nrow(searchGrid), nrow(searchGrid)), ], g)

kfcv <- function(i = k) {
  
  fold <- sample(cut(1:nrow(df), breaks = i, labels = FALSE))
  
  cv_error <- foreach (i = 1:k, .combine = 'cbind') %do% {
    
    train <- df[fold != i, ] 
    test <- df[fold == i, -1]
    
    mapply(function(x, y) {
      
      model <- ctree(Y ~ ., alpha = x, maxdepth = y, data = train)
      
      Kappa(factor(predict(model, test)), df$Y[fold == i])
      
    }, x = searchGrid$alpha, y = searchGrid$maxdepth)
    
  }
  
  return(rowMeans(cv_error))
  
}

tic(); cv_error <- cbind(searchGrid, kappa = cv_parallel(c, kfcv())); toc()

head(cv_error[rev(order(cv_error$kappa)), ])             # minimum error

ct.opt <- cv_error[which.max(cv_error$kappa), ]          # optimal hyperparameters

#########################################################################################

### Parameter Tuning: Support Vector Machine (Linear Kernel)

searchGrid <- expand.grid(kernel = "linear", 
                          cost = c(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.04, 0.05, 0.08, 0.1, 1, 5),
                          epsilon = c(0.001, 0.01, 0.1, 0.5))

searchGrid <- head(searchGrid[sample(1:nrow(searchGrid), nrow(searchGrid)), ], g)

kfcv <- function(i = k) {
  
  fold <- sample(cut(1:nrow(df), breaks = i, labels = FALSE))
  
  cv_error <- foreach (i = 1:k, .combine = 'cbind') %do% {
    
    train <- df[fold != i, ] 
    test <- df[fold == i, -1]
    
    mapply(function(x, y, z) {
      
      model <- svm(Y ~ .,
                   data = train,
                   kernel = x,
                   cost = y, 
                   epsilon = z)
      
      Kappa(factor(predict(model, test)), df$Y[fold == i])
      
    }, x = searchGrid$kernel, y = searchGrid$cost, z = searchGrid$epsilon)
    
  }
  
  return(rowMeans(cv_error))
  
}

tic(); cv_error <- cbind(searchGrid, kappa = cv_parallel(c, kfcv())); toc()

head(cv_error[rev(order(cv_error$kappa)), ])          # minimum error

svml.opt <- cv_error[which.max(cv_error$kappa), ]     # optimal hyperparameters

#########################################################################################

### Parameter Tuning: Support Vector Machine (Polynomial Kernel)

searchGrid <- expand.grid(kernel = "polynomial", 
                          degree = 1:5, 
                          coef0 = c(0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10),
                          cost = c(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1),
                          epsilon = c(0.001, 0.01, 0.1))

searchGrid <- head(searchGrid[sample(1:nrow(searchGrid), nrow(searchGrid)), ], g)

kfcv <- function(i = k) {
  
  fold <- sample(cut(1:nrow(df), breaks = i, labels = FALSE))
  
  cv_error <- foreach (i = 1:k, .combine = 'cbind') %do% {
    
    train <- df[fold != i, ] 
    test <- df[fold == i, -1]
    
    mapply(function(v, w, x, y, z) {
      
      model <- svm(Y ~ .,
                   data = train,
                   kernel = v,
                   degree = w, 
                   coef0 = x,
                   cost = y,
                   epsilon = z)
      
      Kappa(factor(predict(model, test)), df$Y[fold == i])
      
    }, v = searchGrid$kernel, w = searchGrid$degree, x = searchGrid$coef0, 
    y = searchGrid$cost, z = searchGrid$epsilon)
    
  }
  
  return(rowMeans(cv_error))
  
}

tic(); cv_error <- cbind(searchGrid, kappa = cv_parallel(c, kfcv())); toc()

head(cv_error[rev(order(cv_error$kappa)), ])               # minimum error

svmp.opt <- cv_error[which.max(cv_error$kappa), ]          # optimal hyperparameters

#########################################################################################

### Parameter Tuning: Support Vector Machine (Radial Kernel)

searchGrid <- expand.grid(kernel = "radial", 
                          cost = c(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.04, 0.05, 0.08, 0.1, 1, 5),
                          gamma = c(0.001, 0.005, 0.1, 0.5, 1, 2, 3, 4))

searchGrid <- head(searchGrid[sample(1:nrow(searchGrid), nrow(searchGrid)), ], g)

kfcv <- function(i = k) {
  
  fold <- sample(cut(1:nrow(df), breaks = i, labels = FALSE))
  
  cv_error <- foreach (i = 1:k, .combine = 'cbind') %do% {
    
    train <- df[fold != i, ] 
    test <- df[fold == i, -1]
    
    mapply(function(x, y, z) {
      
      model <- svm(Y ~ .,
                   data = train,
                   kernel = x,
                   cost = y, 
                   gamma = z)
      
      Kappa(factor(predict(model, test)), df$Y[fold == i])
      
    }, x = searchGrid$kernel, y = searchGrid$cost, z = searchGrid$gamma)
    
  }
  
  return(rowMeans(cv_error))
  
}

tic(); cv_error <- cbind(searchGrid, kappa = cv_parallel(c, kfcv())); toc()

head(cv_error[rev(order(cv_error$kappa)), ])          # minimum error

svmr.opt <- cv_error[which.max(cv_error$kappa), ]     # optimal hyperparameters

# best kernel (radial, linear, polynomial)

svm.list <- list(svmr.opt, svml.opt, svmp.opt)

svm.opt <- svm.list[[which.max(sapply(svm.list, `[`, c("kappa")))]]

svm.opt # best support vector machine in the discovery set

#########################################################################################
################################ ENSEMBLE EXPERIMENTS ###################################
#########################################################################################

k <- 5;      # number of folds in k-fold cross validation
c <- 40;     # number of times k-fold CV is repeated

# initialize vectors

cycles.ct  <- c(); CV.ct  <- NULL; # conditional inference tree
cycles.knn <- c(); CV.knn <- NULL; # k-nearest neighbors
cycles.rf  <- c(); CV.rf  <- NULL; # random forest
cycles.xt  <- c(); CV.xt  <- NULL; # extremely randomized trees
cycles.xgb <- c(); CV.xgb <- NULL; # extreme gradient boosting
cycles.gp  <- c(); CV.gp  <- NULL; # gaussian processes
cycles.qda <- c(); CV.qda <- NULL; # quadratic discriminant analysis
cycles.lda <- c(); CV.lda <- NULL; # linear discriminant analysis
cycles.nn  <- c(); CV.nn  <- NULL; # neural network
cycles.glm <- c(); CV.glm <- NULL; # logistic regression
cycles.en  <- c(); CV.en  <- NULL; # elastic net
cycles.svm <- c(); CV.svm <- NULL; # support vector machine
cycles.nb  <- c(); CV.nb  <- NULL; # naive bayes classifier
cycles.fda <- c(); CV.fda <- NULL; # flexible discriminant analysis with MARS
cycles.ens <- c(); CV.ens <- NULL; # voting ensemble

x <- matrix(list(), nrow = c, ncol = k) # cross-validation error

tictoc::tic(); for (j in 1:c) {
  
  fold <- sample(cut(1:nrow(df), breaks = k, labels = FALSE))
  
  for (i in 1:k){
    
    cat("Fold", i, "Cycle", j, "\n")
    
    # partition into k folds
    train <- df[fold != i, ]
    test  <- df[fold == i, -1]
    
    # conditional inference trees
    ctree <- ctree(Y ~ ., alpha = ct.opt$alpha, maxdepth = ct.opt$maxdepth, data = train)
    CV.ct[[i]] <- Kappa(predict(ctree, test), df$Y[fold == i])
    
    # random forest
    rf <- ranger(Y ~ . , mtry = rf.opt$mtry, max.depth = rf.opt$max.depth, 
                 data = train, num.trees = 300)
    CV.rf[[i]] <- Kappa(predict(rf, test)$predictions, df$Y[fold == i])
    
    # extremely randomized trees
    xt <- ranger(Y ~ . , mtry = xt.opt$mtry, max.depth = xt.opt$max.depth, 
                 data = train, num.trees = 300, 
                 splitrule = "extratrees", replace = F, sample.fraction = 1)
    CV.xt[[i]] <- Kappa(predict(xt, test)$predictions, df$Y[fold == i])
    
    # XGBoost
    train_xgb <- sparse.model.matrix(Y ~ 0 + . , data = train)
    test_xgb <- sparse.model.matrix(Y ~ 0 + . , data = df[fold == i, ])
    
    params <- list(gamma = 0,
                   booster = "gbtree",
                   objective = "binary:logistic",
                   eta = xgb.opt["eta"],
                   subsample = xgb.opt["subsample"],
                   colsample_bytree = xgb.opt["colsample_bytree"],
                   colsample_bynode = xgb.opt["colsample_bynode"],
                   max_depth = xgb.opt["max_depth"])
    
    xgb <- xgboost(data = train_xgb, label = as.numeric(train$Y) - 1, 
                   params = params, nrounds = xgb.opt["trees"], verbose = FALSE)
    
    CV.xgb[[i]] <- Kappa(factor(round(predict(xgb, test_xgb))), df$Y[fold == i])
    
    # knn
    train_knn <- ohe(standardize(df))[fold != i, ]
    test_knn <- ohe(standardize(df))[fold == i, ]
    
    knn <- knn(train_knn, test_knn, cl = train$Y, k = knn.opt$neighbors)
    CV.knn[[i]] <- Kappa(knn, df$Y[fold == i])
    
    # quadratic discriminant analysis
    qda <- qda(Y ~ . , data = train)
    CV.qda[[i]] <- Kappa(predict(qda, test)$class, df$Y[fold == i])
    
    # linear discriminant analysis
    lda <- lda(Y ~ . , data = train)
    CV.lda[[i]] <- Kappa(predict(lda, test)$class, df$Y[fold == i])
    
    # logistic regression
    glm <- glm(Y ~ ., data = train, family = "binomial", control = list(maxit = 1000))
    CV.glm[[i]] <- Kappa(factor(round(predict(glm, test, type = "response"))), df$Y[fold == i])
    
    # neural network
    train_nn <- ohe.data.frame(standardize(df))[fold != i, ]
    test_nn <- ohe.data.frame(standardize(df))[fold == i, -1]
    
    nn <- neuralnet(Y ~ . , data = train_nn, hidden = nn.opt$hidden, algorithm = nn.opt$algorithm,
                    rep = 1, stepmax = 1e5, linear.output = FALSE, threshold = 0.3)
    CV.nn[[i]] <- Kappa(factor(round(predict(nn, test_nn)[ , 2])), df$Y[fold == i])
    
    # elastic net
    train_en <- ohe(df)[fold != i, ]
    test_en <- ohe(df)[fold == i, ]
    
    en <- glmnet(train_en, train$Y, alpha = en.opt$alpha, lambda = en.opt$lambda,
                 family = "binomial")
    CV.en[[i]] <- Kappa(factor(round(predict(en, test_en, type = "response"))), df$Y[fold == i])
    
    # support vector machine
    if (svm.opt$kernel == "linear") {
      svm <- svm(Y ~ . , data = train, kernel = svm.opt$kernel, cost = svm.opt$cost, 
                 epsilon = svm.opt$epsilon)
    } else if (svm.opt$kernel == "radial") {
      svm <- svm(Y ~ . , data = train, kernel = svm.opt$kernel, cost = svm.opt$cost, 
                 gamma = svm.opt$gamma)
    } else if (svm.opt$kernel == "sigmoid") {
      svm <- svm(Y ~ . , data = train, kernel = svm.opt$kernel, cost = svm.opt$cost, 
                 gamma = svm.opt$gamma, coef0 = svm.opt$coef0)
    } else if (svm.opt$kernel == "polynomial") {
      svm <- svm(Y ~ . , data = train, kernel = svm.opt$kernel, cost = svm.opt$cost, 
                 degree = svm.opt$degree, coef0 = svm.opt$coef0, epsilon = svm.opt$epsilon)
    }
    CV.svm[[i]] <- Kappa(predict(svm, test), df$Y[fold == i])
    
    # naive bayes
    nb <- naiveBayes(Y ~ . , data = train)
    CV.nb[[i]] <- Kappa(predict(nb, test), df$Y[fold == i])
    
    # flexible discriminant analysis (MARS)
    fda <- fda(Y ~ ., data = train, method = mars)
    CV.fda[[i]] <- Kappa(predict(fda, test), df$Y[fold == i])
    
    # gaussian processes
    gp <- gausspr(Y ~ ., data = train, kpar = list(sigma = 0.1), 
                  type = "classification")
    CV.gp[[i]] <- Kappa(predict(gp, test), df$Y[fold == i])
    
    # all predictions    
    x[[j, i]] <- data.frame(ct = predict(ctree, test),
                            knn = knn,
                            gp = predict(gp, test),
                            rf = predict(rf, test)$predictions,
                            xt = predict(xt, test)$predictions,
                            xgb = factor(round(predict(xgb, test_xgb))),
                            qda = predict(qda, test)$class,
                            lda = predict(lda, test)$class,
                            nn = factor(round(predict(nn, test_nn)[ , 2])),
                            glm = factor(round(predict(glm, test, type = "response"))),
                            en = factor(round(predict(en, test_en, type = "response"))),
                            svm = predict(svm, test),
                            nb = predict(nb, test),
                            fda = predict(fda, test),
                            actual = df$Y[fold == i])
    
    # (post-hoc) simple majority vote ensemble with three of the best base learners
    ens <- data.frame(x[[j, i]]$rf, x[[j, i]]$svm, x[[j, i]]$en)
    mv <- factor(apply(ens, 1, function(x) names(which.max(table(x)))))
    CV.ens[[i]] <- Kappa(mv, df$Y[fold == i])
    
    x[[j, i]] <- data.matrix(x[[j, i]]) - 1
    
  }
  
  cycles.ct  <- c(cycles.ct,  mean(CV.ct ))
  cycles.knn <- c(cycles.knn, mean(CV.knn))
  cycles.rf  <- c(cycles.rf,  mean(CV.rf ))
  cycles.xt  <- c(cycles.xt,  mean(CV.xt ))
  cycles.xgb <- c(cycles.xgb, mean(CV.xgb))
  cycles.qda <- c(cycles.qda, mean(CV.qda))
  cycles.lda <- c(cycles.lda, mean(CV.lda))
  cycles.glm <- c(cycles.glm, mean(CV.glm))
  cycles.nn  <- c(cycles.nn,  mean(CV.nn ))
  cycles.en  <- c(cycles.en,  mean(CV.en ))
  cycles.svm <- c(cycles.svm, mean(CV.svm))
  cycles.nb  <- c(cycles.nb,  mean(CV.nb ))
  cycles.fda <- c(cycles.fda, mean(CV.fda))
  cycles.gp  <- c(cycles.gp,  mean(CV.gp ))
  cycles.ens <- c(cycles.ens, mean(CV.ens))
  
  #  beepr::beep()
}; beepr::beep(3); tictoc::toc()

# format convenient for visualization
all <- data.frame(kappa = c(cycles.fda, cycles.nb, cycles.ct, cycles.en, cycles.svm, 
                            cycles.ens, cycles.knn, cycles.nn, cycles.rf, cycles.xt, 
                            cycles.glm, cycles.lda, cycles.qda, cycles.xgb, cycles.gp),
                  model = rep(c("fda", "nb", "ct", "en", "svm", "ens", "knn", "nn",
                                "rf", "xt", "glm", "lda", "qda", "xgb", "gp"), 
                              each = c))

all$model <- with(all, reorder(model, 1 - kappa, median))

aggregate(kappa ~ model, all, mean) # mean accuracy

boxplot(all$kappa ~ all$model, outline = FALSE, xlab = "Model", ylab = "Accuracy",
        main = paste("Model Performance Across", c, "Cycles of k-Fold CV"))

n <- ncol(x[[1, 1]]) - 1 # number of models

for(i in 1:(n + 1)){
  levelProportions <- summary(all$model)/nrow(all)
  values <- all[all$model == levels(all$model)[i], "kappa"]
  jitter <- jitter(rep(i, length(values)), amount=levelProportions[i]/0.8)
  points(jitter, values, pch = 19, col = rgb(1, 0, 0, .3), cex = 0.5) 
}

#########################################################################################
########################### EVALUATE ALL POSSIBLE ENSEMBLES #############################
#########################################################################################

voters <- do.call(expand.grid, replicate(n, list(0:1)))[-1, ]

voters <- voters[apply(voters, 1, function(x) sum(x) %% 2 != 0), ]

voters[voters == 0] <- NA

tictoc::tic()

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

scores <- rowMeans(sapply(1:c, function(l){
  
  rowMeans(sapply(1:k, function(j){
    
    foreach(i = 1:nrow(voters), .combine=rbind, .packages = "caret", 
            .export = c("Kappa", "x", "l", "k", "n", "voters", "df")) %dopar% {
              Kappa(factor(ifelse(apply(data.frame(mapply(`*`, data.frame(x[[l, j]][ , -(n + 1)]), as.numeric(voters[i, ]))),
                                        1, function(x){mean(x, na.rm = TRUE)}) > 0.5, 1, 0)), factor(x[[l, j]][ , n + 1])) * 
                (nrow(x[[1, j]])/nrow(df))/(1/k)
            }}))}))

stopCluster(cl)
tictoc::toc() # 4696.182 sec elapsed

names(voters) <- colnames(x[[1, 1]][ , -(n + 1)])

# best model

scores[which.max(scores)] # over 96% accuracy

voters[which.max(scores), ] # best ensemble

# best models

cbind.data.frame(voters[head(order(scores, decreasing = TRUE), 30), ],  # good choices
                 kappa = head(sort(scores, decreasing = TRUE), 30))

# worst models

cbind.data.frame(voters[head(order(scores), 10), ],
                 kappa = head(sort(scores), 10))

#########################################################################################
########################### LOOKING AT RESULTS FOR WDBC DATA ############################
#########################################################################################

# best models that do not include elastic net or svm

new.mat <- scores[is.na(voters$en) & is.na(voters$svm)]

voters.without.en <- subset(voters, is.na(voters$en) & is.na(voters$svm)) # 95% accuracy

cbind.data.frame(1:30,
                 voters.without.en[head(order(new.mat, decreasing = TRUE), 30), ],
                 kappa = head(sort(new.mat, decreasing = TRUE), 30))

# ensemble that has the best performance increase relative to its base learners

base <- aggregate(kappa ~ model, subset(all, model != "ve"), mean)
base <- base[match(colnames(voters), base$model), ]
base <- matrix(rep(base$kappa, nrow(voters)), nrow = nrow(voters), byrow = TRUE)

diff <- cbind.data.frame(base*voters, scores, 
                         100*(scores/apply(base*voters, 1, max, na.rm = TRUE) - 1))
colnames(diff) <- c(colnames(voters), "kappa", "percent_diff")

head(round(diff[order(diff$percent_diff, decreasing = TRUE), ], 2), 10)

# One ensemble -- conditional inference tree, linear discriminant analysis, and logistic 
# regression -- achieves kappa = 0.93, a 2.56% increase over the best base learner.

head(round(diff[order(diff$percent_diff), ], 3), 10)

# One ensemble -- conditional inference tree, elastic net, and naive bayes --
# achieves kappa = 0.912, a 4.37% decrease over the best base learner.

hist(subset(diff, percent_diff != min(percent_diff))$percent_diff, 
     xlim = c(-4.4, 2.6), breaks = 25, main = "Most Voting Ensembles Do Worse, Not Better",
     xlab = "Percent Increase/Decrease in Performance Over the Best Base Learner", 
     col = "darkblue",
     ylab = "Number of Voting Ensembles")

abline(v = 0, col = "red", lwd = 1, lty = 2)

#########################################################################################