# Implementation from scratch in python(keep updating):
  ## 1. Perceptron: 
  <br /> trained with 70 samples, reached 99.6% validation accuracy with 30 samples.
  ## 2. K-nearest Neighbors (KD-Tree):
  <br /> this is a kd-tree implementation which is a high efficiency implementation of K-nearest Neighbors algorithm.
  ### training data:
  <br /><img width="621" alt="image" src="https://user-images.githubusercontent.com/83719401/160525678-4ec98b78-fe5c-4c2b-b0f5-c2454d09c4f8.png">
  ### classification result with target [9, 5], Num_neighbors = 3:
  <br /><img width="428" alt="image" src="https://user-images.githubusercontent.com/83719401/160525796-48b18c91-7513-458c-9a09-8fcd118f7d71.png">

  ## 3. NaiveBeyes:
  ### main procedure:
  <br />1. Using Maximum_Likelihood to compute the prior probability(introduce Laplacian smoothing to avoid zero prior probability): 
  <br /> <img width="324" alt="image" src="https://user-images.githubusercontent.com/83719401/160288157-cb6f83e0-1e86-4774-93f7-24170b8a87d0.png">
  <br />2. Using prior Matrix with size(1, num_cls) to store prior probability.
  <br />3. Using Maximum_Likelihood to compute the posterior probability(introduce Laplacian smoothing to avoid zero posterior probability):
  <br /> <img width="457" alt="image" src="https://user-images.githubusercontent.com/83719401/160288235-93060ebc-19ac-406f-adea-0cea496e7ad9.png">
  <br />4. Using posterior Matrix with size(num_dim, max_num_val, num_cls) to store posterior probability
  <br />5. Inference by multiplying these probabilities we got from the prior and posterior matrix.
  <br /> <img width="424" alt="image" src="https://user-images.githubusercontent.com/83719401/160288348-2d333362-8d75-4cfc-9e8a-c42bcf253ba7.png">
  ### training data:
  <br /> <img width="671" alt="image" src="https://user-images.githubusercontent.com/83719401/160289186-048ff6ea-7404-4f6d-b53f-bff355202fe5.png">
  
  ## 4. Logistic Regression
  <br /> trainined with 70 samples by SGD(pick one sample at a time to do gradient decent, Overall time complexity (epoch * NUM_sample)), validated with 30 samples and reached 100% validation accuracy.
  <br /> below is the decision boundry W^T * X = 0:
  <br /> <img width="483" alt="image" src="https://user-images.githubusercontent.com/83719401/160667466-78d155f3-6a0c-4823-9d61-7362ff92ecbe.png">



   

# other project(final project from CSC546 intro to machinelearning)
stochastic grid search: tuning hyperparameters of randomforest by combining grid search and random search, that is, randomly sample hyperparameters from a grid in the hyperparameter space. 
