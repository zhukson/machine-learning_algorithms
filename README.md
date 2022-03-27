# Implementation from scratch: (in python)
  1. Perceptron: trained with 70 samples, reached 99.6% validation accuracy with 30 samples.
  2. NaiveBeyes: using Laplacian smoothing to avoid zero conditional probability.
  - Using Maximum_Likelihood to compute the prior probability(introduce Laplacian smoothing to avoid zero prior probability):
  - <img width="324" alt="image" src="https://user-images.githubusercontent.com/83719401/160288157-cb6f83e0-1e86-4774-93f7-24170b8a87d0.png">
  - Using prior Matrix with size(1, num_cls) to store prior probability.
  - Using Maximum_Likelihood to compute the posterior probability(introduce Laplacian smoothing to avoid zero posterior probability):
  - <img width="457" alt="image" src="https://user-images.githubusercontent.com/83719401/160288235-93060ebc-19ac-406f-adea-0cea496e7ad9.png">
  - Using posterior Matrix with size(num_dim, max_num_val, num_cls) to store posterior probability
  - Inference by multiplying these probabilities get from prior and posterior matrix.
  * <img width="424" alt="image" src="https://user-images.githubusercontent.com/83719401/160288348-2d333362-8d75-4cfc-9e8a-c42bcf253ba7.png">

   

# other project(final project from CSC546 intro to machinelearning)
stochastic grid search: tuning hyperparameters of randomforest by combining grid search and random search, that is, randomly sample hyperparameters from a grid in the hyperparameter space. 
