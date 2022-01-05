## Table of contents
1. [Tuning Process](#tuning_process)
2. [Using an Appropriate Scale to pick Hyperparameters](#appropriate_scale)
3. [Normalizing Activations in a Network](#normalizing_activations)
4. [Fitting Batch Norm into a Neural Network](#fitting_batch_norm)


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
