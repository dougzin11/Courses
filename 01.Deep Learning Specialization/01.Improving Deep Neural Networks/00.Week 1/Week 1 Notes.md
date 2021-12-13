## Table of contents
1. [Train / Dev / Test sets](#train_dev_test_set)
2. [Bias / Variance](#bias_variance)
3. [Basic Recipe for Machine Learning](#basic_recipe)
4. [Regularization](#regularization)
5. [Why Regularization Reduces Overfitting](#why_regularization_reduces_overfitting)


## Train / Dev / Test sets <a name="train_dev_test_set"></a>
- The available data is usually breakdown into 3 different sets:
  - Train set: as the name suggests, it is used to train the model
  - Dev set (also called development set or hold-out cross validation): it is used to define the best hyperparameters for the model
  - Test set: it is used to evaluate the final model (already tuned based on the Dev set), providing an unbiased assessment
- It important that the Train, Dev and Test set come from the same distribution (otherwhise, you will be training/tuning your model based on a scenario that is totally different than what you encounter in the Test set)


## Bias / Variance <a name="bias_variance"></a>
- Generally speaking, there are 4 possible scenarios that might happen when evaluating the performance of your model:
  1. High Variance (overfitting scenario): the model is trained too well on the training set and it was not capable of generalizing. Ex: Train set error of 1% and Dev set error of 11%
  2. High Bias (underfitting scenario): the model was not capable of trully learning the relationships that exist in the training set. Ex: Train set error of 15% and Dev set error of 16% (considering a optimal/Bayes of error of ~0%)
  3. High Bias and High Variance: model that performs bad in both the training and dev set. Ex: Train set error of 15% and Dev set error of 30%
  4. Low Bias and Low Variance: the model is capable of learning the relationships that exist in the training set as well as generalizing these relationships. Ex: Train set error of 0.5% and 1% of error in the Dev set


## Basic Recipe for Machine Learning <a name="basic_recipe"></a>
1. For cases where the model has High Bias: we should focus on the training set
  - Try bigger networks (i.e. increase the number of hidden layers, hidden units, etc.)
  - Try to train for longer time
  - Try to find a more appropriate Neural Network architecture
2. For cases where the model has High Variance: we should focus on the dev set
  - Try to get more data
  - Try regularization
  - Try to find a more appropriate Neural Network architecture


## Regularization <a name="regularization"></a>
- As mentioned in the section above, Regularization is one of the techniques that we can use in order to reduce High Variance (e.g. reduce overfitting)
- Regularization for Logistict Regression:
  - The default cost function that we try to minimize through Gradient Descent is given by ```J(W,b) = (1/m) * sum(L(y(i), y'(i)))```
  - When adding Regularization, we include new terms to the cost function:
    - ```J(w,b) = (1/m) * sum(L(y(i), y'(i))) + (lambda/2m) * sum(||w||^2)``` (called L2 regularization)
    - ```J(w,b) = (1/m) * sum(L(y(i), y'(i))) + (lambda/2m) * sum(||w||)``` (called L1 regularization)
    - where:
      - ```lambda``` is called the regularization parameter
      - ```||w||^2 = sum(|w[j]|^2) = w.T * w```, where ```j``` goes from ```1``` to ```n_x``` (number of input features)
      - ```||w|| = sum(|w[j]|)```, where ```j``` goes from ```1``` to ```n_x``` (number of input features)
- The L1 regularization version makes a lot of ```W``` values become zeros, which makes the model size smaller
- Regularization for Neural Network:
  - The default cost function that we try to minimize through Gradient Descent is given by ```J(W1,b1...,WL,bL) = (1/m) * sum(L(y(i),y'(i)))```
  - L2 regularization: ```J(W,b) = (1/m) * sum(L(y(i),y'(i))) + (lambda/2m) * sum((||W||^2)```
  - L1 regularization: ```J(W,b) = (1/m) * sum(L(y(i),y'(i))) + (lambda/2m) * sum((||W||)```
  - where:
    - ```||W[l]||^2 = sum_i(sum_j(w[i, j]^2))``` where ```i``` goes from ```1``` to ```n[l]``` and ```j``` goes from ```1``` to ```n[l-1]```
    - ```||W[l]|| = sum_i(sum_j(w[i, j]))``` where ```i``` goes from ```1``` to ```n[l]``` and ```j``` goes from ```1``` to ```n[l-1]```
- L2 sometimes is also called weight decay. The reason for that is as follows:
  - When using the default cost function, we have that ```dW[l] = (1/m)dZ[l]A[l-1].T```
  - When using L2 regularization, we have that ```dW[l] = (1/m)dZ[l]A[l-1].T + lambda/m * W[l]```
  - So when updating the value of ```W```, we have:
  ```
  W[l] = W[l] - learning_rate * dW[l]
  W[l] = W[l] - learning_rate * ((1/m)dZ[l]A[l-1].T + lambda/m * W[l])
  W[l] = W[l] - learning_rate * (1/m)dZ[l]A[l-1].T - ((learning_rate * lambda)/m) * W[l]
  W[l] = (1 - (learning_rate * lambda)/m) * W[l] - learning_rate * (1/m)dZ[l]A[l-1].T
  ```
  We can see that we are basically taking every element of the matrix ```W[l]``` and multiplying by ```1 - (learning_rate * lambda)/m)``` (which is lower than 1). This is why L2 is also called weight decay
  

## Why Regularization Reduces Overfitting <a name="why_regularization_reduces_overfitting"></a>
