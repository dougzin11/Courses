# Notation used along the course:
- ```m_train``` corresponds to the number of training examples
- ```m_test``` corresponds to the number of test examples
- ```X``` is a matrix defined by taking the training set inputs ```x(1)```, ```x(2)```, etc. and stacking them in columns
- ```Y``` is equal to the label of the training inputs ```y(1)```, ```y(2)```, etc. and stacking them in columns


# Logistic Regression
- Learning algorithm used in binary classification problems (output is either 0 or 1)
- In other words, the algorithm calculates: ```y_hat = P(y = 1 | x)```
- In order to do that, this is done by setting ```y_hat = sigmoid(W.Tx + b)```, where ```x``` is the input matrix, and ```W``` and ```b``` are parameters.
- The sigmoid function is used to limit the output between 0 and 1


# Logistic Regression Cost Function
- Loss function frequently used:  ```L(y_hat, y) = -(ylog y_hat + (1-y)log(1 - y_hat))```
  - This is usually the chosen loss function since it gives a convex optimization problem (local optimum corresponds to global optimum)
- The goal is to minimize the above loss function. There are 2 scenarios we need to study to understand why the above function works:
  - ```y = 1```: ```L(y_hat, y) = -log y_hat``` -> to minimize the loss function, ```y_hat``` has to be as large as possible. However, due to the sigmoid function (remember that ```y_hat = sigmoid(W.Tx + b)```), ```y_hat``` largest value is equal to 1
  - ```y = 0```: ```L(y_hat, y) = -log(1 - y_hat)``` -> to minimize the loss function, ```y_hat``` has to be as low as possible. However, due to the sigmoid function, ```y_hat``` lowest value is equal to 0
- The Loss function is defined with respect to a single training example. On the other hand, the Cost function measures how well the learning algorithm is doing across the entire training set
- Cost function: ```J(W,b) = (1/m)Sum(L(y_hat(i), y(i)))``` where ```(i)``` corresponds to a single training example


# Gradient Descent
