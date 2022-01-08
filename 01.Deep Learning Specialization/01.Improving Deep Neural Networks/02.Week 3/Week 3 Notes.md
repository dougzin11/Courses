## Table of contents
1. [Tuning Process](#tuning_process)
2. [Using an Appropriate Scale to pick Hyperparameters](#appropriate_scale)
3. [Normalizing Activations in a Network](#normalizing_activations)
4. [Fitting Batch Norm into a Neural Network](#fitting_batch_norm)
5. [Why does Batch Normalization work?](#why_does_batch_norm_work)
6. [Batch Normalization at test time](#batch_norm_at_test_time)
7. [Softmax Regression](#softmax_regression)


# Hyperparameter Tuning 
## Tuning Process <a name="tuning_process"></a>
- One of the most challenging aspects of training deep neural networks is to tune the vast number of hyperparameters
- In earlier generations of machine learning algorithms, it was common practice to sample points in a grid and systematically explore all the combinations within the grid. In deep learning, it is recommended to choose the points at random for each hyperparameter (see image below taken from the course slides, where the left grid corresponds to the old practice and the right grid corresponds to what it is usually done in deep learning)

![Screen Shot 2021-12-29 at 11 46 03](https://user-images.githubusercontent.com/36196866/147673973-a5752fd9-d208-4937-9b05-23413b4f45a3.png)

- You can use Coarse to fine sampling scheme:
  - When you find some hyperparameters values that give you better performance, you can zoom into a smaller region around these values and sample more densely within this space


## Using an Appropriate Scale to pick Hyperparameters <a name="appropriate_scale"></a>
- It is important to pick the appropriate scale to search your hyperparameters: depending on the hyperparameter, it is better to randomly search the hyperparameter in a different scale (e.g. logarithm scale)
- Taking the momentum `Beta` used to calculate the exponentially weighted averages (in the RMSProp Optimization, for example) as an example:
  - We usually search `Beta` between the values of `0.9` and `0.999`. However, the change in `Beta` can have different effects depending on which part of the `0.9` and `0.999` scale your `Beta` is:
    1. `Beta = 0.9` corresponds to approximately averaging the last 10 values
    2. `Beta = 0.9005` corresponds to approximately averaging the last 10 values
    3. `Beta = 0.999` corresponds to approximately averaging the last 1000 values
    4. `Beta = 0.9995` corresponds to approximately averaging the last 2000 values
  - As we can see above, we are applying a change of `0.0005` in both examples `1-2` and `3-4`. However, the effect of the same change is extremely different. In these cases, we search the hyperparameters in a different scale (e.g. logarithmic)


# Batch Normalization
## Normalizing Activations in a Network <a name="normalizing_activations"></a>
- Batch Normalization basically normalizes the output `Z[l]` (for a given layer `l`) to help the next layer `l+1` train more efficiently
- Logic for Batch Normalization:
```
Given Z[l] = [z(1), ..., z(m)], i = 1 to m (for each input):

  Compute mean = 1/m * sum(z[i])
  Compute variance = 1/m * sum((z[i] - mean)^2)
  Then Z_norm[i] = (z[i] - mean) / np.sqrt(variance + epsilon) (add epsilon for numerical stability if variance = 0)
  Then Z_tilde[i] = gamma * Z_norm[i] + beta #allow different values of mean and variance
```
- `gamma` and `beta` are learnable parameters (just like `W` and `b`)


## Fitting Batch Norm into a Neural Network <a name="fitting_batch_norm"></a>
- The figure below shows how Batch Normalization works in a Neural Network:
<img width="746" alt="Screen Shot 2022-01-08 at 14 31 36" src="https://user-images.githubusercontent.com/36196866/148653842-6f47fe12-42b1-4e5c-ba31-3cc8fc3e6e87.png">

- It is possible to see that the Batch Normalization is implemented after the calculation of `Z[l]` and the activation function uses the normalized version of the `Z[l]` instead of its original value
- In addition, we can see that the parameters of the Neural Network become:
  - `W[1]`, ..., `W[L]`, `beta[1]`, `gamma[1]`, ..., `beta[L]`, `gamma[L]` (we will see below why `b[1]`, ..., `b[L]` are not parameters when applying Batch Normalization)
- In practice, Batch normalization is usually applied mini-batches of the training set. This means that:
  - The mean is calculated using only the training examples inside of the correspondent mini-batch
  - The variance is calculated using only the training examplesi nside of the correspondent mini-batch
- One important note is that when implementing Batch Normalization:
  - `b[1]`, ..., `b[L]` will be subtracted out by the the Batch Normalization (since we subtract the mean). So adding any constant to all of the examples within a mini-batch, it does not change anything because any constant added will get canceled out by the mean subtraction step
- `beta[l]` and `gamma[l]` will have the exactly same shape of `Z[l]`: `(n[l], m)`


## Why does Batch Normalization work? <a name="why_does_batch_norm_work"></a>
- Batch Normalization reduces the amount that the distribution of the hidden unit values shifts around (i.e. reduces the internal covariate shift that affects the hidden units):
  - Hidden units have their values changing all the time (since the parameters `W` and `b` are updating during the training phase)
  - However, Batch Normalization limits the amount to which updating the parameters `W` and `b` of the previous hidden layers can affect the distribution of values that the next hidden layer sees (since it restricts the distribution by applying constraints on the mean and variance)
  - As a consequence, Batch Normalization causes the hidden units values to become more stable: so even as the earlier layers keep learning, the amount that these changes in distribution force the later layers to adapt is weaken
  - Finally, we can see that Batch Normalization weakens the coupling between the earlier layers and the later layers, allowing each layer to learn more independently of other layers, which in turn speeds up the learning process
- Batch Normalization also produces a regularization effect:
  - Each mini batch is scaled by the mean/variance computed of that mini-batch
  - This adds some noise to the values `Z[l]` within that mini batch. So similar to dropout it adds some noise to each hidden layer's activations
  - This has a slight regularization effect
  - The bigger the size of the mini-batch, the less noise you add to the values of `Z[l]` and, consequently, the less regularization effect you have


## Batch Normalization at test time <a name="batch_norm_at_test_time"></a>
- When we train a NN with Batch normalization, we compute the mean and the variance of the mini-batch
- During testing, however, we might need to process examples one at a time (i.e. we don't have a mini-batch) and yhe mean and the variance of one example does not make sense
- So during test time, we come up with a separate estimate of the mean and variance:
  - We can estimate using exponentially weighted averages across the mini-batches

# Multi-class Classification
## Softmax Regression <a name="softmax_regression"></a>
- In multi-class classification problems, the number of units in the output layers corresponds the same number of classes:
  - For example, if `C = number of classes = 4`, then the output layer has `Ny = C = 4` hidden units
- Each hidden unit in the output layer returns the probability for each of the `C` classes. The standard model for getting your network to do this uses what's called a Softmax layer in the output layer:
```
t = e^(Z[L]) # element-wise exponentiation
y_hat = A[L] = e^(Z[L]) / sum(t)
```

