## Table of contents
1. [Tuning Process](#tuning_process)
2. [Using an Appropriate Scale to pick Hyperparameters](#appropriate_scale)


## Tuning Process <a name="tuning_process"></a>
- One of the most challenging aspects of training deep neural networks is to tune the vast number of hyperparameters
- In earlier generations of machine learning algorithms, it was common practice to sample points in a grid and systematically explore all the combinations within the grid. In deep learning, it is recommended to choose the points at random for each hyperparameter (see image below taken from the course slides, where the left grid corresponds to the old practice and the right grid corresponds to what it is usually done in deep learning)

![Screen Shot 2021-12-29 at 11 46 03](https://user-images.githubusercontent.com/36196866/147673973-a5752fd9-d208-4937-9b05-23413b4f45a3.png)

- You can use Coarse to fine sampling scheme:
  - When you find some hyperparameters values that give you better performance, you can zoom into a smaller region around these values and sample more densely within this space


## Using an Appropriate Scale to pick Hyperparameters <a name="appropriate_scale"></a>
- 
