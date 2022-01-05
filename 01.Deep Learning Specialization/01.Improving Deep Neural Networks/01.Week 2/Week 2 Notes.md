## Table of contents
1. [Mini Batch Gradient Descent](#mini_batch_gradient_descent)
2. [Understanding Mini Batch Gradient Descent](#understanding_mini_batch_gradient_descent)
3. [Exponentially Weighted Averages](#exponentially_weighted_averages)
4. [Bias Correction in Exponentially Weighted Averages](#bias_correction_exponentially_weighted_averages)
5. [Gradient Descent with Momentum](#gradient_descent_momentum)
6. [RMSprop - Root Mean Square Prop](#rmsprop)
7. [Adam Optimization](#adam_optimization)
8. [Learning Rate Decay](#learning_rate_decay)
9. [The Problem of Local Optima](#problem_of_local_optima)


# Optimization Algorithms
## Mini Batch Gradient Descent <a name="mini_batch_gradient_descent"></a>
- Mini batch gradient descent is a variation of the "traditional" Gradient Descent (also called Batch Gradient Descent) that splits the training set into small groups (called batches) that are used to train the model
- Therefore, we have the following differences between  **Batch Gradient Descent** and **Mini Batch Gradient Descent**:
  - **Batch Gradient Descent**: uses the entire training set in 1 iteration
  - **Mini Batch Gradient Descent**: uses 1 batch in 1 iteration. This means that when the model has gone through the entire training set , it will have made several iterations/updates. This is the reason why **Mini Batch Gradient Descent** train models at a much faster pace than **Batch Gradient Descent**
- When a model goes through the entire training set, we say that it has completed 1 **epoch**. Therefore, in **Batch Gradient Descent** 1 epoch means 1 iteration while in **Mini Batch Gradient Descent** 1 epoch means several iterations


## Understanding Mini Batch Gradient Descent <a name="understanding_mini_batch_gradient_descent"></a>
- In **Batch Gradient Descent**, the cost function decreases on each iteration (assuming learning rate is correctly set). However, on **Mini Batch Gradient Descent** this is not true: the cost function generally has a downward trend, but it will be much noisier and even increase in some iterations
- Defining the mini batch size:
  1. If mini batch size is equal to ```m_train```: you are actually implementing **Batch Gradient Descent**
  2. If mini batch size is equal to 1: you are implement what is called **Stochastic Gradient Descent**. In this case, each training example is its own mini batch
  3. If mini batch size is between 1 and ```m_train```: you are implementing the **Mini Batch Gradient Descent**
- The above 3 Gradient Descent algorithms have their own advantages and disadvantages, as shown below (information were also sourced from [machinelearningmastery](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/))

| Gradient Descent Algorithm  | Advantages | Disadvantages |
| ----------------------------| ---------- |  -------------| 
| Batch Gradient Descent      | - The decreased update frequency results in a more stable error gradient and may result in a more stable convergence on some problems | - Commonly, batch gradient descent is implemented in such a way that it requires the entire training dataset in memory and available to the algorithm <br> - Model updates, and in turn training speed, may become very slow for large datasets <br> - The more stable error gradient may result in premature convergence of the model to a less optimal set of parameters|
| Stochastic Gradient Descent | - The frequent updates immediately give an insight into the performance of the model and the rate of improvement <br> - The increased model update frequency can result in faster learning on some problems <br> - The noisy update process can allow the model to avoid local minima (e.g. premature convergence)| - The frequent updates can result in a noisy gradient signal, which may cause the model parameters and in turn the model error to jump around (have a higher variance over training epochs) <br> - Updating the model so frequently is more computationally expensive than other configurations of gradient descent, taking significantly longer to train models on large datasets (we cannot benefit from Vectorization)|
| Mini Batch Gradient Descent | - The model update frequency is higher than batch gradient descent which allows for a more robust convergence, avoiding local minima <br> - The batching allows both the efficiency of not having all training data in memory and algorithm implementations | - Mini-batch requires the configuration of an additional “mini-batch size” hyperparameter for the learning algorithm |


## Exponentially Weighted Averages <a name="exponentially_weighted_averages"></a>
- General equation:
```
V(theta) = Beta * V(theta-1) + (1 - Beta) * theta
```
where `V(theta-1)` corresponds to the equation above applied to the step `theta-1` and theta corresponds to the current value (e.g. today's temperature)
- The equation above is calculating the exponentially weighted average across ~`(1/(1-Beta))` days. Therefore, when:
  - `Beta = 0.9`, the formula above is averaging across the ~ last 10 days
  - `Beta = 0.98` the formula above is averaging across the ~ last 50 days
  - `Beta = 0.5` the formula above is averaging across the ~ last 2 days


## Bias Correction in Exponentially Weighted Averages <a name="bias_correction_exponentially_weighted_averages"></a>
- The bias correction helps make the exponentially weighted averages more accurate, specially at the initial phase of your estimate (where the estimator is underestimating the actual numbers)
-  We have the following formula when implementing the bias correction:
```
V(theta) = ((Beta * V(theta-1) + (1 - Beta) * theta)/(1 - Beta^theta)
```
- It is possible to notice that when `theta` becomes larger (i.e. we are no longer at the initial phase of the estimation), `(1 - Beta^theta)` becomes close to 1 (since `Beta < 0`)


## Gradient Descent with Momentum <a name="gradient_descent_momentum"></a>
- The following block of pseudo code shows how to implement Gradient Descent with Momentum:
```
VdW = 0, Vdb = 0 (initialize the values equal to zero)
on iteration t:
	Compute dW and db     
	
	VdW = Beta * VdW + (1 - Beta) * dW
	Vdb = Beta * Vdb + (1 - Beta) * db
  
  # Update the parameters W and b
	W = W - learning_rate * VdW (instead of using dW we use Vdw)
	b = b - learning_rate * Vdb (instead of using db we use Vdb)
```


## RMSprop - Root Mean Square Prop <a name="rmsprop"></a>
- The following block of pseudo code shows how to implement RMSprop:
```
SdW = 0, Sdb = 0 (initialize the values equal to zero)
on iteration t:
	Compute dW and db
	
	SdW = (Beta * SdW) + (1 - Beta) * dW^2  # squaring is element-wise
	Sdb = (Beta * Sdb) + (1 - Beta) * db^2  # squaring is element-wise
  
  # Update the parameters W and b
	W = W - learning_rate * dW / (sqrt(SdW) + epsilon) # epsilon is added for numerical stability (e.g. avoid division by zero)
	b = B - learning_rate * db / (sqrt(Sdb) + epsilon) # epsilon is added for numerical stability (e.g. avoid division by zero)
```


## Adam Optimization <a name="adam_optimization"></a>
- Adam optimization combines RMSprop and Gradient Descent with Momentum together
- The following block of pseudo code shows how to implement Adam Optimization:
```
VdW = 0, Vdb = 0, SdW = 0, Sdb = 0 (initialize the values equal to zero)
on iteration t:
	Compute dW and db          
	
  # Gradient Descent with Momentum
	VdW = (Beta1 * VdW) + (1 - Beta1) * dW
	Vdb = (Beta1 * Vdb) + (1 - Beta1) * db
	
  # RMSprop
	SdW = (Beta2 * SdW) + (1 - Beta2) * dW^2
	Sdb = (Beta2 * Sdb) + (1 - Beta2) * db^2
	
  # Bias correction
	VdW_corrected = VdW / (1 - Beta1^t)
	Vdb_corrected = Vdb / (1 - Beta1^t)
			
	SdW_corrected = SdW / (1 - Beta2^t)
	Sdb_corrected = Sdb / (1 - Beta2^t)
	
  # Update parameters
	W = W - learning_rate * VdW_corrected / (sqrt(SdW_corrected) + epsilon)
	b = b - learning_rate * Vdb_corrected / (sqrt(Sdb_corrected) + epsilon)
```
- Hyperparameters to tune: `Beta1`, `Beta2`, `learning_rate`, `epsilon`


## Learning Rate Decay <a name="learning_rate_decay"></a>
- It corresponds to the process of slowly reducing the `learning_rate` over time
- Reducing the `learning_rate` over time allows the model to take much bigger steps during its initial phase (when the algorithm is far away from the local optima) and take small steps when getting closer to the local optima (avoiding oscillations around the optimum point)
- One way to implement learning rate decay is to set the `learning_rate = (1 / (1 + decay_rate * epoch_num)) * learning_rate_0`, where:
	- `learning_rate_0` is the initial `learning_rate`
	- `epoch_num` is the number of epoch the algorithm is currently in
	- `decay_rate` is a hyperparameter to tune
- Other learning rate decay implementations:
	- `learning_rate = (0.95 ^ (epoch_num)) * learning_rate_0`
	- `learning_rate = (k / sqrt(epoch_num)) * learning_rate_0`


## The Problem of Local Optima <a name="problem_of_local_optima"></a>
- The concept of local optima in low dimensions is not likely to occur when implementing deep neural networks. This is because for a point to be considered local optima, all the dimensions need to have a convex like function and the chance of that happening in deep neural networks (when the number of dimensions is extremely high) is very low
- In high dimensions, it is much more likely to get stuck in plateaus: regions where the derivative is close to zero for a long time
	- This is where optimization algorithms like momentum, RMSprop or Adam can help you get out of plateaus
