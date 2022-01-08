## Table of contents
1. [Train / Dev / Test sets](#train_dev_test_set)
2. [Bias / Variance](#bias_variance)
3. [Basic Recipe for Machine Learning](#basic_recipe)
4. [Regularization](#regularization)
5. [Dropout Regularization](#dropout_regularization)
6. [Why does drop-out work](#why_dropout_work)
7. [Other Regularization Methods](#other_regularization_methods)
8. [Normalizing Inputs](#normalizing_inputs)
9. [Vanishing / Exploding Gradient](#vanishing_exploding_gradient)
10. [Weight Initialization for Deep Networks](#weight_initialization)


# Setting up your Machine Learning Application
## Train / Dev / Test sets <a name="train_dev_test_set"></a>
- The available data is usually breakdown into 3 different sets:
  - Train set: as the name suggests, it is used to train the model
  - Dev set (also called development set or hold-out cross validation): it is used to define the best hyperparameters for the model
  - Test set: it is used to evaluate the final model (already tuned based on the Dev set), providing an unbiased assessment
- It important that the Train, Dev and Test set come from the same distribution (otherwise, you will be training/tuning your model based on a scenario that is totally different than what you encounter in the Test set)


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


# Regularizing your Neural Network
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
  

## Dropout Regularization <a name="dropout_regularization"></a>
- The dropout regularization eliminates neurons (and its ingoing and outgoing weights) based on a probability. The dropout is done for each iteration of gradient descent
- How to implement dropout ("Inverted Dropout", which is the most common dropout technique used):
```
Define keep_prob (probability to keep a neuron: 0 <= keep_prob <= 1) for a certain layer l
d[l] = np.random.rand(a[l].shape[0], a[l].shape[1]) < keep_prob
a[l] = np.multiply(d[l], a[l])
a[l] /= keep_prob # we divide by keep_prob to avoid reducing the expected value of a[l] due to the dropout
```
- `d[l]` is used to define what neurons to zero out both in the forward and backward propagation steps
- When predicting the output, we don't use dropout (otherwise we would have random output, generating noise to our predictions). Since we scaled the values of ```a[l]``` by dividing the original value by ```keep_prob```, we ensured that the expected values during the test time don't change even if we don't use dropout


## Why does drop-out work? <a name="why_dropout_work"></a>
- Intuition: the network can't rely on one single feature (because any of the features/neurons could go away at random due to the dropout regularization). It needs to spread out the weights
- If you're more worried about some layers overfitting than others, you can set a lower `keep_prob` for them. However, this gives you even more hyperparameters to search for when using cross-validation


## Other Regularization Methods <a name="other_regularization_methods"></a>
1. Data Augmentation
- When you increase the amount of data by adding slightly modified copies of the original data (e.g. flipping an image horizontaly)
- The copies, however, would never be as good as new independent data

2. Early stopping
- First, we monitor the performance of the algorithm on both the training and validation set
- We then stop the training when the performance on the validation set starts to degradate
- This technique tries to simultaneously minimize the cost function as well as not overfit (which contradicts the orthogonalization principle - i.e. try to solve one problem at a time)


# Setting up your Optimization Problem
## Normalizing Inputs <a name="normalizing_inputs"></a>
- We usually normalize the inputs in order to speed up the training process: when we don't normalize, the ```W``` parameters (e.g. ```W1```, ```W2```, etc.) can have values on totally different scales, leading to a cost function with an enlogated shape which, in turn, makes the optimization through gradient descent slow
- Normalizing the inputs allow the ```W``` parameters to have values on a similar scale, leading to a cost function with a consistent shape which, in turn, makes the optimization through gradient descent much faster
- When normalizing the test set, the variance and mean should come from the training set
- For more information about why normalization helps Neural Networks, Timo Stöttner has written a nice article about it [here](https://towardsdatascience.com/why-data-should-be-normalized-before-training-a-neural-network-c626b7f66c7d)


## Vanishing / Exploding Gradient <a name="vanishing_exploding_gradient"></a>
- The Vanishing / Exploding gradients occurs when your derivatives become very small or very big. This difficults training the model:
  - Very small derivatives lead to a long time for the model to train 
  - Very big derivatives lead to big updates which can make the model unstable and unable to learn the data
- Carefully choosing how to initialize the weights partially solve this problem (but not completely)


## Weight Initialization for Deep Networks <a name="weight_initialization"></a>
- So lets say when we initialize ```W``` like this (better to use with tanh activation) - called Xavier Initialization:
```
np.random.rand(shape) * np.sqrt(1/n[l-1])
```
- Or we can initialize ```W``` like this (better to use with ReLU activation):
```
np.random.rand(shape) * np.sqrt(2/n[l-1])
```
- The idea behind these initializations is to initialize the weights such that the variance of the activations are the same across every layer. This constant variance helps prevent the gradient from exploding or vanishing
