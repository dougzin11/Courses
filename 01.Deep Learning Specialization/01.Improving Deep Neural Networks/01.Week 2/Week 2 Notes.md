## Table of contents
1. [Mini Batch Gradient Descent ](#mini_batch_gradient_descent)
2. [Understanding Mini Batch Gradient Descent](#understanding_mini_batch_gradient_descent)


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
| Stochastic Gradient Descent | - The frequent updates immediately give an insight into the performance of the model and the rate of improvement <br> - The increased model update frequency can result in faster learning on some problems <br> - The noisy update process can allow the model to avoid local minima (e.g. premature convergence)| - The frequent updates can result in a noisy gradient signal, which may cause the model parameters and in turn the model error to jump around (have a higher variance over training epochs) <br> - Updating the model so frequently is more computationally expensive than other configurations of gradient descent, taking significantly longer to train models on large datasets (it cannot leverage from Vectorization)|
| Mini Batch Gradient Descent | - The model update frequency is higher than batch gradient descent which allows for a more robust convergence, avoiding local minima <br> - The batching allows both the efficiency of not having all training data in memory and algorithm implementations | - Mini-batch requires the configuration of an additional “mini-batch size” hyperparameter for the learning algorithm |


## Exponentially Weighted Averages <a name="exponentially_weighted_averages"></a>
- 

