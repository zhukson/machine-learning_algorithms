# Implementation from scratch in python(keep updating):
  ## 0. Linear Regression
  <br /> W = [w1, w2 ..., wn, 1] X = [x1, x2, x3 ..., xm, 1] xi = [xi_1, xi_2, ..., xi_n] where 1 stands for bias b.
  <br /> loss function:
  <br /><img width="415" alt="image" src="https://user-images.githubusercontent.com/83719401/160980297-52e2bd3d-8620-47a3-9acf-3e05865cdc8a.png">
  <br /> gradient_decent:
  <br /><img width="217" alt="image" src="https://user-images.githubusercontent.com/83719401/160980145-7e7c2ce8-79eb-47b3-8b4b-ed0eeb0eedf2.png">
  ### sample result:
  <br /> <img width="300" alt="image" src="https://user-images.githubusercontent.com/83719401/160980024-10604942-b557-4b9d-88eb-3d16100e5b6c.png">

  ## 1. Perceptron:
  <br /> loss function:
  <br /> <img width="345" alt="image" src="https://user-images.githubusercontent.com/83719401/160980786-a1365709-f269-4d9f-b610-d8f675648c14.png">
  <br /> step 1:    find all samples satisfied y(wx+b) < 0(which means the inference result is not correct)
  <br /> step 2:    do gradient decent:
  <br /> <img width="186" alt="image" src="https://user-images.githubusercontent.com/83719401/160980520-5179544b-e1d9-4230-ac30-75590465e86e.png">
  <br /> <img width="155" alt="image" src="https://user-images.githubusercontent.com/83719401/160980541-704f25b9-7960-47f6-bb81-7185231886d4.png">
  <br /> step 3:    repeat step 1 - 2 until there are no error classification samples or reached maximum iterations.
  ### decision boundry sample:
  <br /> <img width="250" alt="image" src="https://user-images.githubusercontent.com/83719401/160784890-70e33b0e-cbfd-4796-b5dd-556d6d1a7494.png">
  <img width="250" alt="image" src="https://user-images.githubusercontent.com/83719401/160790427-4d78939d-c893-4968-b505-f0704eff4e68.png">

  ## 2. K-nearest Neighbors (KD-Tree):
  <br /> this is a kd-tree implementation which is a high efficiency implementation of K-nearest Neighbors algorithm.
  ### training data:
  <br /><img width="350" alt="image" src="https://user-images.githubusercontent.com/83719401/160981478-6ed4cc8b-0bee-4eeb-bd3d-48d295e242d0.png">
  ### classification result with target [9, 5], Num_neighbors = 3:
  <br /><img width="350" alt="image" src="https://user-images.githubusercontent.com/83719401/160525796-48b18c91-7513-458c-9a09-8fcd118f7d71.png">

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
  <br /> <img width="350" alt="image" src="https://user-images.githubusercontent.com/83719401/160667466-78d155f3-6a0c-4823-9d61-7362ff92ecbe.png">



   

# other project(final project from CSC546 intro to machinelearning)
stochastic grid search: tuning hyperparameters of randomforest by combining grid search and random search, that is, randomly sample hyperparameters from a grid in the hyperparameter space. 
