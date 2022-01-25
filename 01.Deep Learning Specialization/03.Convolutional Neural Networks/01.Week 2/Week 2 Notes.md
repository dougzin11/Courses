## Table of contents
1. [Classic Networks](#classic_networks)
2. [ResNets](#res_nets)
3. [Why ResNets Work](#why_res_nets_work)
4. [Network in Network and 1x1 Convolutions](#network_in_network)
5. [Transfer Learning](#transfer_learning)
6. [Data Augmentation](#data_augmentation)
7. [State of Computer Vision](#state_of_computer_vision)


# Case studies
## Classic Networks <a name="classic_networks"></a>
- **LeNet-5**
  - The goal for this model was to identify handwritten digits in a `32x32x1` gray image
  - Below you can find the architecture for the model **(image taken from the course)**:

![Screen Shot 2022-01-23 at 14 08 28](https://user-images.githubusercontent.com/36196866/150689660-66285dac-ab67-4d86-b11e-ab02b276eb30.png)

  - As you go deeper in the network, the height (`n_h`) and width (`n_w`) tends to decrease while the number of channels tends to increase (`n_c`)
  - Activation functions were `sigmoid` or `tanh`
- **AlexNet**
  - Below you can find the architecture for the model **(image taken from the course)**:

![Screen Shot 2022-01-23 at 14 15 04](https://user-images.githubusercontent.com/36196866/150689902-ee596d1e-61a6-4827-a3c4-ff5e505c638e.png)

  - Similar to LeNet-5, but much bigger
  - Activation function was `ReLu`
- **VGG-16**
  - Created with the idea of having a much simpler network:
    - Conv layers: always `3x3` filters and `s=1` and `same padding`
    - Pooling layers: alwayys `max-pooling`, size of `2x2` and stride of `s=2`
- Below you can find the architecture for the model:

<img width="912" alt="Screen Shot 2022-01-23 at 14 25 05" src="https://user-images.githubusercontent.com/36196866/150690288-d4729268-bcea-49cf-9ac4-2e42f7401156.png">

- Name **VGG-16** comes from the fact that it has 16 layers with weights

## ResNets <a name="res_nets"></a>
- Very, very deep neural networks are difficult to train because of vanishing and exploding gradient types of problems
- In this section, we talk about how to skip connections which allows you to take the activation from one layer and suddenly feed it to another layer even much deeper in the neural network
- ResNets are built out of something called Residual block:
  - The image below tries to give a brief overview of what these Residual blocks do **(image taken from the course)**:

<img width="947" alt="Screen Shot 2022-01-23 at 14 44 08" src="https://user-images.githubusercontent.com/36196866/150690971-b9a50e4e-1bda-4a11-b3b7-3d938997682e.png">

  - In summary:
    - Following the "traditional" way of calculating in a neural network, we would walk through the "main path" in the image above. This means that for `a[l]` to affect `a[l+2]`, it would have to go through many intermediary steps
    - With residual blocks we create a shortcut where we eliminate these intermediary steps, allowing `a[l]` to influence `a[l+2]` directly
- ResNets:
  - They are composed of many Residual blocks stacked together to form a deeper network. Below you can find a simple illustration **(image taken from the course)**:

![Screen Shot 2022-01-23 at 14 42 20](https://user-images.githubusercontent.com/36196866/150690892-25a8b308-f1f9-41d2-addf-544918ce6c03.png)

  - The ResNets can go deeper without hurting the performance. Normal NN (called Plain networks in the original paper), in theory, would get better performance when they get deeper and deeper. However, due to the vanishing and exploding gradients problems, the performance of the network suffers as it goes deeper **(image taken from the course)**

<img width="923" alt="Screen Shot 2022-01-23 at 14 44 46" src="https://user-images.githubusercontent.com/36196866/150691001-7ef98153-37ca-4d98-9e38-79254476c093.png">


## Why ResNets Work <a name="why_res_nets_work"></a>
- In order to understand why ResNets work, let us work through an example:
  - Imagine that we have a big NN as follows:
  
  ```
   X (input) -> Big NN -> a[l]
  ```
 
  - Now, let's take a copy of the NN above and add a Residual block at the end:

  ```
                            ___________________________
                           |                           |
  X (input) -> Big NN -> a[l] -> Layer1 -> Layer2 -> a[l+2]
  ```
  
  - In the network above, `a[l+2] = g(z[l+2] + a[l]) = g(W[l+2]a[l+1] + b[l+2] + a[l])`
  - If we are using weight decay (L2 regularization) to both `W` and `b`, it is possible that `W[l+2]` and `b[l+2]` could be zero. In this scenario, `a[l+2] = g(a[l]) = a[l]`, if `g(.)` is a ReLu activation function (assuming that `a[l]` is greather than 0)
  - This shows that Residual blocks can learn Identify function, which in turns mean that adding the 2 layers at the end doesn't hurt the performance
  - **Important Note**: in the above example, we assumed that `z[l+2]` and `a[l]` have actually the same dimensions. This is one of the reasons why in ResNets there are a lot of `same` paddings (to preserve dimensions). However, in case `z[l+2]` and `a[l]` have different dimensions, we add an extra matrix such as follows: `a[l+2] = g(W[l+2]a[l+1] + b[l+2] + W_s a[l])`. `W_s` can be a fixed matrix that implements zero padding or even a matrix with learnable parameters 


## Network in Network and 1x1 Convolutions <a name="network_in_network"></a>
- Let's consider the below example:
  - Input: `6x6x32`
  - Conv: **`n_c` filters** of the size `1x1x32` 
  - Output: `6x6xn_c`
- We can see that 1x1 convolutions can change the number of channels (increase or decrease). However, it is generally used to decrease the number of channels to reduce computation costs
- It is also possible to use 1x1 convolutions to keep the number of channels. In this case, 1x1 convolutions simply apply non-linearity to the input data, allowing the network to learn more complex functions (e.g. ReLu, tanh, etc.)
- In summary, 1x1 convolutions are useful when:
  - we want to change the number of channels (usually decrease it)
  - we want to add even more non-linearity to our network


## Inception Network Motivation <a name="inception_network_motivation"></a>
- When designing a layer for a ConvNet, you might have to pick: do you want a 1 by 3 filter, or 3 by 3, or 5 by 5, or do you want a pooling layer?
- What the inception network does is it says, why should you do them all?
- Below is a simplified diagram of the Inception layer **(image taken from the course)**:

![Screen Shot 2022-01-24 at 18 24 41](https://user-images.githubusercontent.com/36196866/150867374-36c64884-ac75-4ae3-b65c-41c5cb7ce31d.png)

  - **Important Note**: for the pooling layer to output data on the same dimensions as the other filters, we need to use padding in the pooling layer

- One problem with the Inception layer is the computational cost associated with it:
  - If we take the `5x5` filter from the diagram above we have:
    - For an image of the input size of `28x28x192`, we have `28x28x192x5x5` calculations
    - If the `5x5` convolution should output data of the size `28x28x32`, then in total we have `28x28x192x5x5x32 = 120 million` calculations
  - Using `1x1` convolutions we can drastically reduce the amount of calculations
- Applying `1x1` convolutions to reduce the computational costs:
  - We can apply `1x1` convolutions as an intermediate step to reduce the amount of calculations, as shown below **(image taken from the course)**:
![Screen Shot 2022-01-24 at 18 41 31](https://user-images.githubusercontent.com/36196866/150869609-333a7230-23bf-4dd5-ae2f-6c118f29049a.png)
  - We can see now that the `5x5` filter is applied over an image of the size of `28x28x16` instead of the original `28x28x192`
  - Now, the amount of calculations corresponds to `28x28x16x1x1x192 = 2.5 million` + `28x28x32x5x5x16 = 10 million` = `12.5 million`
  - the `1x1` convolution is also called as **Bottleneck Layer**
- The Inception layer with `1x1` convolutions is shown below **(image taken from the course)**:

<img width="946" alt="Screen Shot 2022-01-24 at 18 57 14" src="https://user-images.githubusercontent.com/36196866/150871789-56112cde-52e0-40cd-87ae-e01dcb6e25a3.png">

- The inception network, more or less, brings together a lot of these modules


## Transfer Learning <a name="transfer_learning"></a>
- If you're building a computer vision application rather than training the ways from scratch, from random initialization, you often make much faster progress if you download weights that someone else has already trained on the network architecture and use that as pre-training and transfer that to a new task that you might be interested in
- To do transfer learning, here is what you could do:
  - Small amount of data available: delete the last layer of the NN (i.e. keep all the other weights fixed) and add a new last layer with weights to be learned during the training process
  - Enough data available: you can keep fewer layers of the original NN (i.e. delete the last 4 layers and not only the last one) and add new layers to be trained
  - A significant amount of data: you can fine-tune all the layers (i.e. do not freeze any layer), but instead of random initializing the parameters, you can take the learned parameters and learn it from there
- When we talk about Transfer Learning, 2 new definitions may come up:
  - Pre-training: corresponds to the initial phase of training on the original task
  - Fine-tuning: corresponds to the training phase on the new task
- Transfer Learning usually makes sense in the following scenarios:
  1. When the original task has the same input as the new task (e.g. original and new task are trying to predict images - not necessarily the same type of images)
  2. You have a lot of data available for the original task you are transferring from and relatively less data for the new task you are transferring to
  3. Low-level features from the original task could help learn the new task
- When doing Transfer Learning, you can speed up the training process even further by doing the following:
  - Pass the input data through the layers that are not trainable (i.e. weights were frozen)
  - Take the output of the last frozen layer and save this to disk
  - This output will feed the following trainable layers
  - This method avoids passing the input data through all the frozen layers in every single epoch, thus speeding up the training process


## Data Augmentation <a name="data_augmentation"></a>
- Data augmentation corresponds to the techniques used to increase the amount of data by adding slightly modified copies of already existing data
- Some example of data augmentation methods **(images taken from the course)**:
  - Mirroring
  - Random cropping

![Screen Shot 2022-01-25 at 11 23 37](https://user-images.githubusercontent.com/36196866/150994981-900b287e-1ec9-464b-b224-0545365c7bc2.png)
  
  - Rotation
  - Sharing
  - Local warping
  - Color shifting: this makes your algorithm more robust to changes in the color of the images. To decide how much distortion to add to RGB channels, you can use **PCA Color Augmentation**
![Screen Shot 2022-01-25 at 11 22 34](https://user-images.githubusercontent.com/36196866/150994462-c92cae21-3b02-4633-9276-4d37e5059858.png)
- When applying data augmentation, it is important to be careful: the augmented data needs to preserve what the true label is
  - Example: rotating an image by 180 degrees may add noise to your model if you are trying to predict numbers. The number `6`, for example, might turn into `9` when rotating the image (but the true label will be assigned as `6`), which in turn can hurt your model performance


## State of Computer Vision <a name="state_of_computer_vision"></a>
- The more data available, generally, the less hand engineering needs to be done
- Some techniques used for doing well on benchmarks/winning competitions:
  1. Ensembling: train several networks independently and average their **outputs**
  2. Multi-crop at test time:
    - Run the model on multiple versions of the test set (e.g. 10 crop technique: 5 cropped images of the original version + 5 cropped images of the mirror version)
    - Average the output of the model on the multiple versions of the test set
