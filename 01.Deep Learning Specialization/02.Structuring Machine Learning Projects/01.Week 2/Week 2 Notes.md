## Table of contents
1. [Carrying out error analysis](#carry_error_analysis)
2. [Cleaning up incorrectly labeleded data](#incorrectly_labeled_data)
3. [Training and Testing on different distributions](#training_testing_different_distributions)


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

