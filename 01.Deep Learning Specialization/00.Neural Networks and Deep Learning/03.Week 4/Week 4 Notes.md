## Table of contents
1. [Deep L-layer Neural Network](#deep_l_layer_nn)
2. [Forward Propagation in a Deep Network](#forward_propagation)
3. [Getting your Matrix Dimensions Right](#matrix_dimensions)
4. [Building Blocks of Deep Neural Networks](#building_blocks)
5. [Forward and Backward propagation](#forward_backward_propagation)


## Deep L-layer Neural Network <a name="deep_l_layer_nn"></a>
- We will use the notation ```L``` to denote the number of layers in a NN
- ```n[l]``` is the number of neurons in a specific layer ```l```:
  - ```n[0]``` denotes the number of neurons in the input layer, ```n[1]``` is the number of neurons in the first hidden layer and ```n[L]``` is the number of neurons in the output layer
- ```a[l]``` is the activation in layer ```l``` where ```a[l] = g[l](z[l])```
- ```W[l]``` is the weights for ```z[l]``` in layer ```l```


## Forward Propagation in a Deep Network <a name="forward_propagation"></a>
- Forward propagation general equation for one training example:
```
z[l] = W[l]a[l-1] + b[l]
a[l] = g[l](z[l])
```
- Forward propagation general equation for ```m_train``` training examples:
```
Z[l] = W[l]A[l-1] + b[l]
A[l] = g[l](Z[l])
```


## Getting your Matrix Dimensions Right <a name="matrix_dimensions"></a>
- ```W[l]``` has ```(n[l], n[l-1])``` dimension
- ```b[l]``` has ```(n[l], 1)``` dimension
- ```dW[l]``` has the same dimension as ```W[l]``` while ```db[l]``` has the same dimension as ```b[l]```
- ```Z[l]``` has ```(n[l], m)``` dimension. Since ```A[l]``` just applies an activation function on ```Z[l]```, then ```A[l]``` has also ```(n[l], m)``` dimension
- ```dZ[l]``` has the same dimension as ```Z[l]``` and ```dA{l]``` has the same dimension as ```A[l]```: ```(n[l], m)```

## Building Blocks of Deep Neural Networks <a name="building_blocks"></a>
- Below you can find a simple diagram of the forward and back propagation steps for a layer l in a Neural Network:
![bb](https://user-images.githubusercontent.com/36196866/144145652-6a4cb138-21a7-45b0-b5a4-4873a7226c98.PNG)

## Forward and Backward propagation <a name="forward_backward_propagation"></a>
- In the forward propagation step, we do the following:
```
Input: A[l-1]
Cache: Z[l], where Z[l] = W[l]A[l-1] + b[l]
Output: A[l], where A[l] = g[l](Z[l])
```
- In the backward propagation step, we do the following:
```
Input: dA[l]
Cache: 
  - dZ[l], where dZ[l] = dA[l]*g[l]'(Z[l]) (element-wise product)
Output: 
  - dW[l], where dW[l] = (1/m)dZ[l]A[l-1].T
  - db[l], where db[l] = (1/m)np.sum(dZ[l], axis=1, keepdims=True)
  - dA[l-1], where dA[l-1] = W[l].TdZ[l]
```

## Parameters vs Hyperparameters

