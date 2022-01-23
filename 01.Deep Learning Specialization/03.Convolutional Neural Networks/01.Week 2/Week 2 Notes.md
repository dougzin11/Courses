## Table of contents
1. [Classic Networks](#classic_networks)
2. [ResNets](#res_nets)
3. [Why ResNets Work](#why_res_nets_work)


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
- 
