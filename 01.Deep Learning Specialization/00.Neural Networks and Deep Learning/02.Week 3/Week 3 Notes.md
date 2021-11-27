## Table of contents
1. [Neural Networks Overview](#neural_networks_overview)
2. [Neural Network Representation](#neural_networks_representation)
3. [Computing a Neural Network's Output](#computing_neural_network)
4. [Vectorizing Across Multiple Examples](#vectorizing_multiple_examples)
5. [Why do you need non-linear activation functions](#non_linear_activation_functions)
6. [Random Initialization](#random_initialization)

## Neural Networks Overview <a name="neural_networks_overview"></a>
- A simple Neural Network (with 2 layers - we don't count the input layer, as we will see later) can look like the image below

![NN_diagram](https://user-images.githubusercontent.com/36196866/142620575-c3de26dd-10ab-4ea1-b685-e54d22ca96c1.PNG)

- Each node in the above diagram corresponds to 2 calculation steps:
  - ```z = W.Tx + b```
  - ```a = sigmoid(z)```

![node_conput](https://user-images.githubusercontent.com/36196866/142727078-2e3108aa-27ba-4ec2-ae9d-5eed35638596.PNG)

- So in the above 2 layer Neural Network, the calculations could be broken down as follow:

![shallow_nn](https://user-images.githubusercontent.com/36196866/142726663-eedfd66b-a953-44bd-85d9-cbc0bb373e37.PNG)


## Neural Network Representation <a name="neural_networks_representation"></a>
- There are 3 different types of layers in a Neural Network:
  - Input layer: corresponds to the first layer (although we don't count it) that brings the initial data into the Neural Network
  - Hidden layer(s): corresponds to the intermadiate layers between the input and output layer (mentioned below). The name convention comes from the fact that the output of the nodes present in these layers are not seen during the training phase
  - Output layer: corresponds to the last layer of the network which is responssible for generating the desired output

![nn_representation](https://user-images.githubusercontent.com/36196866/142727023-b607dc96-e289-4726-ba08-c4e680b100c2.PNG)


## Computing a Neural Network's Output <a name="computing_neural_network"></a>
- As mentioned in the [Neural Networks Overview](#neural_networks_overview) section, each node corresponds to 2 calculation steps.
- So for the hidden units, we can write the computation steps as follows:

![nn_cac](https://user-images.githubusercontent.com/36196866/142727274-d1f7a82a-13a9-47f4-871a-461841ae9815.PNG)

- Vectorizing the equations above we have:
  - ```z[i] = W[i].Tx + b[i]```, where ```[i]``` represents the identification of each layer in the Neural Network the calculation is beign performed
  - ```a[i] = sigmoid(z[i])```, where ```[i]``` represents the identification of each layer in the Neural Network the calculation is beign performed
  - where:
    - ```W[i].T``` is a matrix obtained by stacking the weights ```W``` of each node from layer ```[i]``` as rows (e.g. ```W[1]``` is obtained by stacking ```W1[1]T``` (1st row), ```W2[1]T``` (2nd row), ...)
    - ```b[i]``` is a matrix obtained by stacking the wegiths ```b``` of each node from layer ```[i]``` as rows (e.g. ```b[1]``` is obtained by stacking ```b1[1]``` (1st row), ```b2[1]``` (2nd row), ...)
    - ```a[i]``` is obtained by applying the sigmoid function element-wise in the ```z[i]``` matrix.


## Vectorizing Across Multiple Examples <a name="vectorizing_multiple_examples"></a>
- So far, we learned how to vectorize the equations only for a single training example. Therefore, so far, we can use the equations for all ```m_train``` examples as follows (in the 2 layer neural network shown previously):
```
for i = 1 to m_train:
  z[1](i) = W[1]x(i) + b[1]
  a[1](i) = sigmoid(z[1](i))
  z[2](i) = W[2]a[1](i) + b[2]
  a[2](i) = sigmoid(z[2](i))
```
- Vectorizing across the ```m_train``` training examples, we have:
```
  Z[1] = W[1]X + b[1]
  A[1] = sigmoid(Z[1])
  Z[2] = W[2]A[1] + b[2]
  A[2] = sigmoid(Z[2])
```
where:
  - ```X``` corresponds to the ```m_train``` training examples stacked as columns (e.g. ```x(1)``` as 1st column, ```x(2)``` as 2nd column, ..., ```x(m_train)``` as ```m_train``` column)
  - ```Z[1]``` is obtained by stacking ```z[1](i)``` across all ```m_training``` examples as columns (e.g. ```z[1](1)``` as 1st column, ```z[1](2)``` as 2nd column, ..., ```z[1](m_training)``` as  ```m_training``` column). Therefore, ```Z[1]``` has a shape of ```(number_of_neurons_in_layer_1, m_training)```
  - ```A[1]``` is obtained by stacking ```a[1](i)``` across all ```m_training``` examples as columns (e.g. ```a[1](1)``` as 1st column, ```a[1](2)``` as 2nd column, ..., ```a[1](m_training)``` as ```m_training``` column). Therefore, ```A[1]``` has a shape of ```(number_of_neurons_in_layer_1, m_training)```


## Why do you need non-linear activation functions <a name="non_linear_activation_functions"></a>
- If the network uses the identity function as its activation functions (or alternatively don't have an activation function), then it turns out that the neural network is just outputting a linear function of the input. In this situation, no matter how many layers your network has, all it is doing is just computing a linear activation function (so you might not have any hidden layers)

## Random Initialization <a name="random_initialization"></a>
- For logistic regression, it is okay to initializate the parameters to zero. However, in general, initialization the parameters to zero and applying gradient descent will not work
- In the scenario where all your hidden units have the same weight/influence on the output unit, initializing the parameters to zero may cause the hidden units to be completely identical (i.e. calculating the exact same function) no matter how many iterations you train the network
- The solution to this is to initialize the parameters randomly (i.e. it avoids the symmetry problem mentioned above)
- One important note is that we usually prefer to initialize the parameters to low values: if we are using specific activation functions (e.g. tanh or sigmoid), the gradient values are too small (meaning that the gradient descent - how fast we update the parameters - is going to be slow)
