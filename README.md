# Empirical_Study_of_Ensemble_Learning_Methods

Ensembles are systems of models working together. Under the right conditions, the collection of models ("ensemble") can outperform the individual models ("base learners"). Ensembles have been applied successfully to problems in regression and classification. Various strategies have been proposed, inclduding bagging, boosting, stacked generalization, multiple imputation ensembles, committees of neural networks, and super learners. This project explores simple voting ensembles, and highlights the difficulty of choosing the best ensemble from a set of candidate learners.

I evaluated 14 models for classification:
* Elastic Net
* Support Vector Machine (Choice of Linear, Radial, or Polynomial Kernel)
* Single Hidden Layer Neural Network
* Gaussian Process (Gaussian Kernel)
* Extremely Randomized Trees
* k-Nearest Neighbors
* XGBoost (Gradient Boosted Decision Trees)
* Random Forest
* Flexible Discriminant Analsyis by Optimal Scoring (using MARS)
* Quadratic Discriminant Analysis
* Linear Discriminant Analysis
* Generalized Linear Model (Logistic Regression)
* Conditional Inference Tree
* Naive Bayes

The setting of the experiment was the Wisconsin Diagnostic Breast Cancer data set, a binary classification problem with 30 numeric predictors describing benign and malignant tumor samples. I have also prepared the code to accomodate other binary classification problems easily. I hope to extend the experiment one day to multiclass classification and regression.

The best simple voting ensemble, discovered by repeated k-fold cross-validation, was a combination of elastic net, support vector machine with polynomial kernel, and random forest. This came as a surprise, since these were not the #1, #2, and #3 models, but rather the #1, #2, and #8 models.

![Figure 1](https://github.com/timothygmitchell/Empirical_Study_of_Ensemble_Learning_Methods/blob/main/ModelPerformance.png)

Accuracy was measured using Cohen's kappa, which is appropriate for imbalanced classes. Abbreviations: *ens*, simple voting ensemble; *en*, elastic net; *svm*, support vector machine; *nn*, neural network, *gp*, gaussian process; *xt*, extremely randomized trees ("ExTra trees"); *knn*, k-nearest neighbors; *xgb*, XGBoost; *rf*, random forest; *fda*, flexible discriminant analysis; *qda*, quadratic discriminant analysis; *lda*, linear discriminant analysis; *glm*, generalized linear model (logistic regression); *ct*, conditional inference tree; *nb*, naive bayes.

![Figure 2](https://github.com/timothygmitchell/Empirical_Study_of_Ensemble_Learning_Methods/blob/main/HistEnsemblePerformance.png)

For a set of *N* learners, one can prepare 2^*N* - 1 meaningful combinations -- including standalone base learners and ensembles. If one wishes to avoid ties, it is convenient to consider only the odd-numbered ensembles, which reduces the number by half. I estimated the generalization error for all (2^14 - 1)/2 simple voting ensembles prepared in this manner, and found that most of them did *worse* than the best base learner in the system.

I would like to prepare a stacked generalization model, and I would to see if the population of ensembles has a similar distribution, or if stacking protects the ensemble from error.
