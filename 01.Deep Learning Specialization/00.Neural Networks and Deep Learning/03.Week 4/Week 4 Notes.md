## Table of contents
1. [Deep L-layer Neural Network](#deep_l_layer_nn)
2. [Forward Propagation in a Deep Network](#forward_propagation)


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
