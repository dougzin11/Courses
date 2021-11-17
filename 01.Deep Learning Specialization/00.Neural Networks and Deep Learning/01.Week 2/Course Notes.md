## Table of contents
1. [Notation](#notation)
2. [Logistic Regression](#logistic_regression)
3. [Logistic Regression Cost Function](#logistic_regression_cost_function)
4. [Gradient Descent](#gradient_descent)
5. [Computation Graph](#computation_graph)
6. [Logistic Regression Gradient Descent](#logistic_regression_gradient_descent)

## Notation used along the course <a name="notation"></a>
- ```m_train``` corresponds to the number of training examples
- ```m_test``` corresponds to the number of test examples
- ```X``` is a matrix defined by taking the training set inputs ```x(1)```, ```x(2)```, etc. and stacking them in columns
- ```Y``` is equal to the label of the training inputs ```y(1)```, ```y(2)```, etc. and stacking them in columns


## Logistic Regression <a name="logistic_regression"></a>
- Learning algorithm used in binary classification problems (output is either 0 or 1)
- In other words, the algorithm calculates: ```y_hat = P(y = 1 | x)```
- In order to do that, this is done by setting ```y_hat = sigmoid(W.Tx + b)```, where ```x``` is the input matrix, and ```W``` and ```b``` are parameters.
- The sigmoid function is used to limit the output between 0 and 1


## Logistic Regression Cost Function <a name="logistic_regression_cost_function"></a>
- Loss function frequently used:  ```L(y_hat, y) = -(ylog y_hat + (1-y)log(1 - y_hat))```
  - This is usually the chosen loss function since it gives a convex optimization problem (local optimum corresponds to global optimum)
- The goal is to minimize the above loss function. There are 2 scenarios we need to study to understand why the above function works:
  - ```y = 1```: ```L(y_hat, y) = -log y_hat``` -> to minimize the loss function, ```y_hat``` has to be as large as possible. However, due to the sigmoid function (remember that ```y_hat = sigmoid(W.Tx + b)```), ```y_hat``` largest value is equal to 1
  - ```y = 0```: ```L(y_hat, y) = -log(1 - y_hat)``` -> to minimize the loss function, ```y_hat``` has to be as low as possible. However, due to the sigmoid function, ```y_hat``` lowest value is equal to 0
- The Loss function is defined with respect to a single training example. On the other hand, the Cost function measures how well the learning algorithm is doing across the entire training set
- Cost function: ```J(W,b) = (1/m)Sum(L(y_hat(i), y(i)))``` where ```(i)``` corresponds to a single training example


## Gradient Descent <a name="gradient_descent"></a>
- Algorithm used to learn the parameters ```W``` and ```b``` during training phase. As we will see, ```W``` and ```b``` will be set as the values who minimize the Cost function ```J(W,b)``` (defined above). Just note that, instead of trying to minimze the Loss function ```L(y_hat, y)```, we are actually interested in minimizing the Cost function ```J(W,b)``` (since it is a function that measures the learning algorithm across the whole training set instead of a single training example)
- In a nutshell, this is how the Gradient Descent algorithm works:
  - Initialize ```W``` and ```b``` as 0 (or random values - due to the Loss function chosen above, we have a convex optimization problem, which means that the different initialization values should give similar end results)
  - Calculate derivative of our Cost function ```J(W,b)``` with respect to ```W``` and the derivative of ```J(W,b)``` with respect to ```b``` (```dJ/dW``` and ```dJ/db```, respectively)
  - Update ```W``` and ```b``` as follows:
    - ```W := W - alpha dJ/dW```, where ```alpha``` corresponds to the learning rate
    - ```b := b - alpha dJ/db```, where ```alpha``` corresponds to the learning rate


## Computation Graph <a name="computation_graph"></a>
- Organizes a computation from left-to-right (usually known as foward propagation) and another computation from right-to-left (usually known as back propagation)
- The photo below shows a simple example of the foward propagation step

![foward_propagation](https://user-images.githubusercontent.com/36196866/142196635-f3ff13d2-991d-43b3-806e-3822ae4a3717.PNG)

- The right-to-left computation (back propagation) corresponds to the step where we compute derivative of ```J(W,b)``` (cost function) with respect to any weight in the network
- In the photo where we show an example of the foward propagation, we could calculate the derivatives as follows:
  - ```dJ/dv = 3```
  - ```dv/da = 1```
  - ```dv/du = 1```
  - ```dJ/da = dJ/dv * dv/da = 3 * 1 = 3```
  - ```dJ/du = dJ/dv * dv/du = 3 * 1 = 3```
  - ```dJ/db = dJ/dv * dv/du * du/db = 3 * 1 * c = 3c = 6``` (since ```c=2```)
  - ```dJ/dc = dJ/dv * dv/du * du/dc = 3 * 1 * b = 3b = 9``` (since ```b=3```)


## Logistic Regression Gradient Descent <a name="logistic_regression_gradient_descent"></a>
- The computation graph for logistic regression can be created as the image below:

![logistic_regression_computation_graph](https://user-images.githubusercontent.com/36196866/142284869-5b6e9274-0247-4e0f-a36f-65df127eadca.PNG)

- As we can see in the photo above, the final output is the Loss function (not the Cost function). Therefore, the above image is examplifying the gradient descent for a single training example (in a Neural Network composed of a single Neuron
- The important derivates in the back propagation step can be summarized as follows:
  - ```dL(a, y) = -(y/a) + (1 - y)/(1 - a)```
  - ```dL(a, y)/dz = dL/da * da/dz = [-(y/a) + (1 - y)/(1 - a)] * [a * (1 - a)] = a - y```
  - ```dL(a, y)/dw1 = dL/dz * dz/dw1 = x1*dL/dz = x1(a-y)```
  - ```dL(a, y)/dw2 = dL/dz * dz/dw2 = x2*dL/dz = x2(a-y)```
  - ```dL(a, y)/db = dL/dz * dz/db = dL/dz * 1 = a - y```
- To calculate the Gradient Descent over ```m_train``` examples (i.e. output Cost function), we can see that since ```J(W,b) = (1/m)Sum(L(y_hat(i), y(i)))```, then:
  - ```dJ(W,b)/dW(j) = (1/m)Sum(dL(y_hat(i), y(i))/dW(j))```, here ```(i)``` corresponds to a single training example and ```(j)``` corresponds to a specific parameter (e.g. ```w1```, ```w2```, etc.)
