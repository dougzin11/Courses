## Table of contents
1. [Carrying out error analysis](#carry_error_analysis)
2. [Cleaning up incorrectly labeleded data](#incorrectly_labeled_data)
3. [Training and Testing on different distributions](#training_testing_different_distributions)
4. [Bias and Variance with mismatched data distributions](#bias_variance_mismatched_data)
5. [Addressing Data Mismatch](#addressing_data_mismatch)
6. [Transfer Learning](#transfer_learning)
7. [Multi-task Learning](#multi_task_learning)


# Error Analysis
## Carrying out error analysis <a name="carry_error_analysis"></a>
- Error analysis corresponds to the process of manually examining mistakes that your algorithm is making, giving you insights on what to do next
- See an example scenario below:
  - Imagine that you have a cat classification model with a 10% error on your dev set. You are interested in decreasing this error
  - To know how to best spend your efforts on (i.e. what action will decrease the dev error the most) you perform an error analysis:
    - Get 100 mislabeled dev set examples at random
    - Count how many of the 100 examples are dogs
    - If only a few of them are dogs (e.g. 5 dogs out of the 100 examples), then focusing on making your model do better on dogs will decrease your error to 9.5% in the best scenario
    - However, if a lot of them are dogs (e.g. 50 dogs out of the 100 examples), then focusing on making your model do better on dogs will decrease your error to 5.0% in the best scenario (which sounds much more significant gain)


## Cleaning up incorrectly labeleded data <a name="incorrectly_labeled_data"></a>
- Deep Learning algorithms are quite robust to **random** errors in the **training set**, as long as the percentage of mislabeled data is not high
- However, Deep Learning algorithms are less robust to **systematic** errors in the **training set** (e.g. white dogs consistently labeled as cats may result in the algorithm learning to always classify white dogs as cats)
- For mislabeled cases in the **dev/test set**, it is good to perform an error analysis. For example:

  | Image      | Dog | Great Cats | Blurry | Mislabeled | Comments                             |
  | ---------- | --- | ---------- | ------ | ---------- | ------------------------------------ |
  | ...        |     |            |        |            |                                      |
  | 98         |     |            |        | V          | labeler missed cat in the background |
  | 99         |     | V          |        |            |                                      |
  | 100        |     |            |        | V          | Drawing of a cat; not a real cat     |
  | % of total | 8%  | 43%        | 61%    | 6%         |                                      |
    
  - If the overall dev set error is 10%, then:
    - Errors due to mislabeled data: 0.6%
    - Errors due to other causes: 9.4%
  - Maybe focusing on other causes will reduce the dev set error more significantly than the 0,6% you would end up getting by fixing the incorrect labels
- Imagine now a different scenario:
  - Dev set error is 2%
    - Mislabeled data: 0.6%
    - Classifier A: 1.9% error
    - Classifier B: 2.1% error
    - Here you cannot guarantee that classifier A is better than B since the mislabeled data corresponds to approximately 30% of your dev set error (significant value). In this case, correcting the mislabeled data is an important step to make
- Consider the following guidelines while correcting the dev/test set examples:
  - Apply the same process to your dev and test sets to make sure they continue to come from the same distribution
  - Consider examining examples your algorithm got right as well as ones it got wrong
  - Train and dev/test data may now come from a slightly different distributions (t's very important to have dev and test sets to come from the same distribution. But it could be OK for a train set to come from slightly other distribution)


# Mismatched training and dev/test set
## Training and Testing on different distributions <a name="training_testing_different_distributions"></a>
- Different approaches can be taken when your training set has a distribution different from the your dev/test set:
  1. Shuffle all the data and randomly split the data into train/dev/test sets **(not recommended)**:
    - Advantages: all the data sets (train/dev/test) will come from the same distribution
    - Disadvantages: your dev/test set may not necessarily represent the real word scenario you care about. The problem here is that you would be optimizing your model for a scenario different than you actually want
  2. Make your dev/test set contains only examples that resembles the scenario you care about:
    - Advantages: you are optimizing your model for the scenario you care about
    - Disadvantages: training distribution is different from dev/test set distributions


## Bias and Variance with mismatched data distributions <a name="bias_variance_mismatched_data"></a>
- The Bias and Variance analysis changes when the training distribution is different from dev/test set distributions. For example, imagine that we have the following model performance:
  
  | Error type  | Performance |
  | ----------- | ----------- |
  | Human Error | 0%          |
  | Train Error | 1%          |
  | Dev Error   | 10%         |
  
  - In this example, when we went from the Train Error to the Dev Error two things changed at the same time: (1) algorithm saw data in the training set but not on the dev set (Variance problem) and (2) the distribution of the data in the dev set is different. And because of this, it is hard to tell how much of the 9% difference is because of Variance and how much of it is because of the difference in distributions
  - Therefore, we can no longer safely assume that this is a Variance problem
- To solve this issue, we create a new set called **trainining-dev set**: data set with the same distribution as the training set which is not used for training. To create the **training-dev set** this is what we do:
  - Randomly shuffle the training set
  - Carve out a piece of the training set to be the **training-dev set**
- This way, when we want to analyze the Bias and the Variance, we also evaluate the model performance on the training-dev set. See an example below:

  | Error type      | Performance |
  | --------------- | ----------- |
  | Human Error     | 0%          |
  | Train Error     | 1%          |
  | Train-Dev Error | 9%          |
  | Dev Error       | 10%         |
  
  - We can conclude now that:
    1. 8% (gap between Train error and Train-Dev Error) error corresponds to the Variance (i.e. the only difference between the 2 is the fact that the model saw data from 1 set but not from the other). So we have in this example a Variance problem
    2. 1% (gap between Train-Dev Error and Dev Error) error comes from the fact that we are analyzing performance on different distributions. This is called Data-Mismatch
- So here are the general elements important when dealing with cases where the training set has a different distribution than the dev/test set:
  1. Human-level error (a proxy for Bayes error)
  2. Train error
    - Calculates `avoidable bias = training error - human-level error`
    - If the difference is big then we have an Avoidable bias problem
  3. Train-dev error
    - Calculates `variance = training-dev error - training error`
    - If the difference is big then we have a high variance problem
  4. Dev error
    - Calculates `data mismatch = dev error - train-dev error`
    - If the difference is much bigger then the train-dev error we have a Data mismatch problem
    - - Sometimes this error can be even lower than the Train and/or Train-dev error. One reason could be the fact that the dev set may be easier for the model to predict
  4. Test error
    - Calculates `degree of overfitting to dev set = test error - dev error`
    - If the difference is big (positive) then maybe you need to find a bigger dev set. Since the dev set and test set come from the same distribution, the only way for there to be a huge gap here is if you somehow managed to overfit to the dev set
    - Sometimes this error can be even lower than the Train and/or Train-dev error. One reason could be the fact that the test set may be easier for the model to predict


## Addressing Data Mismatch <a name="addressing_data_mismatch"></a>
- There are not completely systematic solutions to address Data Mismatch, but here is what you could try:
  1. Carry out manual error analysis to try to understand the differences between training and dev/test sets
  2. Once you find how the dev/test set differs from the training set, you can try to find ways to make training data more similar, or collect more data similar to dev/test sets
    - If your goal is to make the training data more similar to your dev set one of the techniques you can use is **Artificial data synthesis**, which consists of combinining some of your training data with something that can convert it to the dev/test set distribution. For example:
      - Training set: clear car audios
      - Dev set: noisy car audios
      - Artifical data synthesis: clear car audios + car noise
    - If the car noise added in the example below corresponds to only a subset of noise that exists in the dev set, we could potentially make the model overfit to this tiny subset of noise (example: only synthesize noisy car audios by repeting the same noisy audio)


# Learning from multiple tasks
## Transfer Learning <a name="transfer_learning"></a>
- One of the most powerful ideas in deep learning is that sometimes you can take knowledge the neural network has learned from one task and apply that knowledge to a separate task. This is called Transfer Learning
  - So for example, maybe you could have the neural network learn to recognize objects like cats and then use that knowledge or use part of that knowledge to help you do a better job reading x-ray scans
- To do transfer learning, here is what you could do:
  1. Small amount of data available: delete the last layer of the NN (i.e. keep all the other weights fixed) and add a new last layer with weights to be learned during the training process
  2. Enough data available: you can keep less layers of the original NN (i.e. delete the last 4 layers and not only the last one) and add new layers to be trained
- When we talk about Transfer Learning, 2 new definitions may come up:
  - Pre-training: corresponds to the initial phase of training on the original task
  - Fine-tuning: corresponds to the training phase on the new task
- Transfer Learning usually makes sense in the following scenarios:
  1. When the original task has the same input as the new task (e.g. original and new task are trying to predict images - not necessarily the same type of images)
  2. You have a lot of data available for the original task you are transferring from and relatively less data for the new task you are transferring to
  3. Low level features from the original task could be helpful for learning the new task


## Multi-task Learning <a name="multi_task_learning"></a>
- 
