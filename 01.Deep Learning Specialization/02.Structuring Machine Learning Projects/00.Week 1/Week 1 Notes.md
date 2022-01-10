## Table of contents
1. [Orthogonalization](#orthogonalization)
2. [Single numnber evaluation metric](#single_number_evaluation_metric)
3. [Satisficing and Optimizing metric](#satisficing_and_optimizing_metric)
4. [Train/dev/test distributions](#train_dev_test_distribution)
5. [When to change dev/test sets and metrics](#when_to_change_dev_test_sets_and_metrics)
6. [Why human-level performance](#why_human_level_performance)
7. [Avoidable bias](#avoidable_bias)
8. [Improving your model performance](#improving_model_performance)


# Introduction to ML Strategy
## Orthogonalization <a name="orthogonalization"></a>
- Orthogonalization is an approach where we try to address problems and issues of our model separately (i.e. without interfering much on other things), making the process of tuning your model much easier
- For supervised learning models to do well, usually there are 4 things that you want to hold:
  1. Fit training set well on cost function
  2. Fit dev set well on cost function
  3. Fit test well on cost function
  4. Model performs well in real-world
- So orthogonalization tries to address the 4 points mentioned above separately. For example:
  - If we want our model to fit better on the training set, we may want to try bigger networks, change our optimization algorithm, etc.
  - If we want our model to fit better on the dev set, we may want to implement regularization, use a bigger training set, etc.
  - However, if we implement early stopping, this is something that interferes both on how well our model fits the training set as well as how well our model fits the dev set (i.e. it is one method that is less orthogonalized)


# Setting up your goal
## Single numnber evaluation metric <a name="single_number_evaluation_metric"></a>
- Setting up a single number evaluation metric instead of several metrics allows you to iterate faster and better understand which models you trained is actually giving the best performance
- See an example below:
  - When looking at the Precision and Recall from classifiers `A` and `B` it is hard to easily know which one has the best performance, and this is because we are comparing more than one single evaluation metric

| Classifier | Precision	| Recall |
| ---------- | --------- | ------ |
| A 	       | 95%	      | 90%    |
| B	         | 98%	      | 85%    |

  - If we combine Precision and Recall in a metric called F1 score (which corresponds to the harmonic mean of Precision and Recall), we can easily see that classifer `A` is doing a better job than classifier `B`

| Classifier | Precision	| Recall | F1 score |
| ---------- | --------- | ------ | -------- |
| A 	       | 95%	      | 90%    | 92.4%    |
| B	         | 98%	      | 85%    | 91.0%    |


## Satisficing and Optimizing metric <a name="satisficing_and_optimizing_metric"></a>
- Sometimes it is hard to get a single number evaluation metric. Ex:

| Classifier | F1 score	| Running Time |
| ---------- | --------- | ------------ | 
| A	         | 90%	     | 80 ms        | 
| B	         | 92%	     | 95 ms        | 
| C          | 92%       | 1,500ms      |

- To solve this, we can choose a single optimizing metric and turn the other metrics as satisficing metrics:
  - F1 score can be the optimizing metric: so we want to optimize this
  - Running time is a satisficing metric: we want this to be below a certain value (e.g. < 100 ms). If it is 80 ms or 95 ms, it does not matter because they are < 100 ms

- As a general rule, if you have `N` evaluation metrics:
  - 1 metric is the optimizing metric
  - `N-1` are satisficing metrics


## Train/dev/test distributions <a name="train_dev_test_distribution"></a> 
- Train, dev and test set have to come from the same distribution
- Dev and Test set have to be chosen in a way that they reflect data you expect to get in the future and consider important to do well on


## When to change dev/test sets and metrics <a name="when_to_change_dev_test_sets_and_metrics"></a>
- If you find that the evaluation metric is not giving the correct rank order preference for what it is actually a better algorithm, then it is time to think about defining a new evaluation metric. See an example below:
  - Imagine that in a cat classification example we have the following results:
    - Metric: classification error
    - Algorithm A: 3% error (but pornographic images are treated as cat images)
    - Algorithm B: 5% error
  - If we choose the algorithm solely based on the metric, we would pick algorithm `A`, which has a major flaw. In this case, we can change our evaluation metric:
    - Old evaluation Metric: `(1/m) * sum(y_pred[i] != y[i])`
    - New evaluation Metric: `(1/sum(w[i])) * sum(w[i] * (y_pred[i] != y[i]))`, where:
      - `w[i] = 1` if `x[i]` is not a pornographic image
      - `w[i] = 10` if `x[i]` is pornographic image
  - This way we are applying a much greater penalization for pornographic images
- Conclusion: if doing well on your metric + dev/test set does not correspond to doing well on your application, change your metric and/or dev/test set


# Comparing to human-level performance
## Why human-level performance <a name="why_human_level_performance"></a>
- We compare to human-level performance because of two main reasons:
  - Because of advances in deep learning, machine learning algorithms are suddenly working much better and so it has become much more feasible in a lot of application areas for machine learning algorithms to actually become competitive with human-level performance
  - It turns out that the workflow of designing and building a machine learning system is much more efficient when you're trying to do something that humans can also do
- However, there is a theoretical limit that a model can never surpass called `Bayes optimal error`. It turns out that for a lot of tasks there isn't much error range between human-level error and Bayes optimal error


## Avoidable bias <a name="avoidable_bias"></a>
- As an example, suppose we have the following model performance of a classifier built to identify whether or not an image contains a cat:

| Scenario | Training Error	| Dev Error | Human Error |
| -------- | -------------- | --------- | ----------- |
| A        | 8%             | 10%       | 1%          |
| B        | 8%             | 10%       | 7.5%        |
  
- In scenario A: the classifier is significantly worse than human-level performance on the training set. In this case, we would focus on reducing bias
- In scenario B: the classifier is close to the human-level performance on the training set, but development error is a bit higher. In this case, we would focus on reducing the variance
- We can follow the 2 lines of thought mentioned above because in a lot of tasks (including computation vision, which is the case mentioned above) we can use human-level error as a proxy for Bayes optimal error
- So having an estimate of human-level performance gives you an estimate of Bayes error, allowing you to more quickly make decisions as to whether you should focus on trying to reduce bias or trying to reduce variance
- When you surpass the human-level performance, you might no longer have a good estimate of Bayes error, making it difficult for you to know which approach you should focus on (i.e. reduce bias or reduce variance)
- 1 new definition may be useful: `Avoidable bias = Training error - Bayes optimal error`, where sometimes we can use the human-level error as a proxy for Bayes optimal error

  
## Improving your model performance <a name="improving_model_performance"></a>
- The two fundamental asssumptions of supervised learning:
  1. You can fit the training set pretty well. This is roughly saying that you can achieve low avoidable bias
  2. The training set performance generalizes pretty well to the dev/test set. This is roughly saying that variance is not too bad
- To improve your deep learning supervised system follow these guidelines:
  1. Look at the difference between human level error and the training error - avoidable bias
  2. Look at the difference between the dev/test set and training set error - Variance.
  3. Decide which of the 2 errors (bias or variance) you should focus on
- If avoidable bias is large you have these options:
  1. Train bigger model
  2. Train longer/better optimization algorithm (like Momentum, RMSprop, Adam)
  3. Find better NN architecture/hyperparameters search
- If variance is large you have these options:
  1. Get more training data
  2. Regularization (L2, Dropout, data augmentation)
  3. Find better NN architecture/hyperparameters search
