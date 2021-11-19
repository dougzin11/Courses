## Table of contents
1. [Neural Networks Overview](#neural_networks_overview)

## Neural Networks Overview <a name="neural_networks_overview"></a>
- A simple Neural Network (with 2 layers - we don't count the input layer, as we will see later) can look like the image below

![NN_diagram](https://user-images.githubusercontent.com/36196866/142620575-c3de26dd-10ab-4ea1-b685-e54d22ca96c1.PNG)

- Each node in the above diagram corresponds to 2 calculation steps:
  - ```z = W.Tx + b```
  - ```a = sigmoid(z)```
- So in the above 2 layer Neural Network, the calculations could be broken down as follow:

TODO: CRIAR IMAGEM AQUI 

## Neural Network Representation
- There are 3 different types of layers in a Neural Network:
  - Input layer: corresponds to the first layer (although we don't count it) that brings the initial data into the Neural Network
  - Hidden layer(s): corresponds to the intermadiate layers between the input and output layer (mentioned below). The name convention comes from the fact that the output of the nodes present in these layers are not seen during the training phase
  - Output layer: corresponds to the last layer of the network which is responssible for generating the desired output

TODO: CRIAR IMAGEM DE UM NETWORK INPUT-HIDDEN-OUTPUT
