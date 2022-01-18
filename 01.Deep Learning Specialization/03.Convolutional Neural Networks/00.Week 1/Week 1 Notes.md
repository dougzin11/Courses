## Table of contents
1. [Computer Vision](#computer_vision)
2. [Edge Detection example](#edge_detection_example)
3. [Padding](#padding)
4. [Strided Convolutions](#strided_convolutions)
5. [Convolution over volumes](#convolution_over_volumes)
6. [One Layer of a Convolutional Network](#one_layer_conv_network)


# Convolutional Neural Networks
## Computer Vision <a name="computer_vision"></a>
- Examples of computer vision problems includes:
  - Image classification
  - Object detection (detect an object and localize them)
  - Neural style transfer (changes the style of an image using another image)
- One of the challenges of computer vision problems is that inputs can get pretty big:
  - In a `64x64x3` image, the input feature has a dimension of `12288`
  - In a `1000x1000x3` image, the input feature has a dimension of `3,000,000`. This means that if in the 1st hidden layer you have `1000` hidden units, then there are `3 billion` parameters for the model to learn just in this 1st hidden layer
    - With this amount of parameters, it is difficult to get enough data to prevent a neural network from overfitting
    - In addition, computational and memory requirements to train a neural network with billion parameters is just infeasible


# Edge Detection example <a name="edge_detection_example"></a>
- The convolution operation is one of the fundamental building blocks of a convolutional neural network. We will see how the convolution operation works using edge detection as the motivating example
- CNNs work by decomposing the image into three scales from low order to high order (taken from Hoss Belyadi, Alireza Haghighat, in [Machine Learning Guide for Oil and Gas Using Python](https://www.sciencedirect.com/book/9780128219294/machine-learning-guide-for-oil-and-gas-using-python), 2021):
  - In the low order, small details or local features of the image like lines, edges, and curves can be extracted by early layers of the neural network
  - Deeper layers would assemble these features
  - Final layers reconstruct the whole image
- Example of a 6x6 grayscale image (i.e. 6x6x1 matrix - no RGB channel):

  | 3 | 0 | 1 | 2 | 7 | 4 |
  | - | - | - | - | - | - |
  | 1 | 5 | 8 | 9 | 3 | 1 |
  | 2 | 7 | 2 | 5 | 1 | 3 |
  | 0 | 1 | 3 | 1 | 7 | 8 |
  | 4 | 2 | 1 | 6 | 2 | 8 |
  | 2 | 4 | 5 | 2 | 3 | 9 |
  
- We can then create a 3x3 matrix (also called a filter or kernel):

  | 1 | 0 | -1 |
  | - | - | -- |
  | 1 | 0 | -1 |
  | 1 | 0 | -1 |
  
- The convolution operation can be calculated as follows:
  - Paste the filter or kernel on top of the 6x6 grayscale image
  - Calculate the element-wise multiplication: record the value in the output matrix
  - Move the filter or kernel until the whole 6x6 image is covered, repeating the element-wise multiplication step
- To calculate the 1st element of the output matrix, we do as follows:
  | 3 x 1| 0 x 0| 1 x -1 | 2 | 7 | 4 |
  | ---- | ---- | ------ | - | - | - |
  | 1 x 1| 5 x 0| 8 x -1 | 9 | 3 | 1 |
  | 2 x 1| 7 x 0| 2 x -1 | 5 | 1 | 3 |

  = 3x1 + 1x1 + 2x1 + 0x0 + 5x0 + 7x0 + 1x-1 + 8x-1 + 2x-1 = -5
  
- After covering the whole 6x6 image, this is what we have:
  
  | -5  | 4   | 0   | 8   |
  | --- | --- | --- | --- |
  | -10 | -2  | 2   | 3   | 
  | 0   | -2  | -4  | -7  | 
  | -3  | -2  | -3  | -16 |


## Padding <a name="padding"></a>
- In the last section we saw that a `6x6` matrix convolved with `3x3` filter/kernel gives us a `4x4` matrix. To give it a general rule:
  - if a matrix `nxn` is convolved with `fxf` filter/kernel give us `n-f+1 x n-f+1` matrix
- From the general rule, we can see that:
  1. Filter/kernels shrink the output (`f` just needs to be greater than 1). This is a problem when you want to build really deep neural networks (the last layer could end up with a shrunk image)
  2. Each corner of the original image (before the application of the filter/kernel) is used only 1 time (so we are throwing away a lot of information present in these corners)
- To solve the above problems mentioned above, you can **pad** the image:
  - This corresponds to the process of adding rows/columns to the image before convolution
  - We will call the padding amount `P` the number of row/columns that we will insert in the top, bottom, left and right of the image (e.g. when we apply a padding of `P` = 1 on an image `3x3`, then we end up with an image of `4x4`)
- When applying padding, the output matrix will have a shape defined by the general rule below:
  - If a matrix `nxn` is convolved with `fxf` filter/kernel and padding, the output matrix will be a `n+2p-f+1 x n+2p-f+1` matrix. Example:
    - `n = 6`, `f = 3`, and `p = 1`: the output matrix will have `n+2p-f+1 = 6+2-3+1 = 6` (the size of the image after convolution remains intact)
- Padding terminologies that are important:
  - `Valid convolutions`: it means no padding
  - `Same convolutions`: it means to apply padding so that the output size is the same as the input size. The padding amount `P` is given by `P = (f-1)/2`
- The `f` value for the filters/kernels are usually odd numbers. There are 2 reasons for that:
  1. If `f` was an even number when using `Same convolutions` we would end up with an asymmetric padding `P`
  2. When `f` is odd, the filter/kernel has a "central" position. It is good to have this central position to use this as a reference of the location of the kernel/filter


## Strided Convolutions <a name="strided_convolutions"></a>
- Stride (defined by the variable `S`) corresponds to the number of rows/columns (in an image it would mean the number of pixels) that we jump when we move the filter/kernel during the convolution process
- When applying stride, the output matrix will have a shape defined by the general rule below:
  - If a matrix `nxn` is convolved with `fxf` filter/kernel, padding `p` and stride `s`, the output matrix will be a `((n+2p-f)/s)+1 x ((n+2p-f)/s)+1` matrix
  - When the `((n+2p-f)/s)+1` is not an integer, we take the floor value (i.e. we round down to the nearest integer)
- In math textbooks, the convolution operation adds one more step: it flips the filter both in the horizontal and vertical axis before actually doing the convolution. However, by convention, in machine learning, we usually do not bother with this flipping operation (when we do not flip the filter, we are technically doing what is called a cross-correlation operation)


## Convolution over volumes <a name="convolution_over_volumes"></a>
- In this section, we will see how to convolve 3D images
- A 3D image has the following properties: `image height`, `image width` and `image # of channels`
- Similarly, our filter/kernel applied in the convolution process will also have the same 3 properties: `filter height`, `filter width` and `filter # of channels`
  - To convolve a 3D image, the number of channels in the image and the filter/kernel has to be the same: `image # of channels` = `filter # of channels`
- Below there is an example (image taken from the course):
  - Input image of `6x6x3`
  - Filter of `3x3x3`
  - Stride `S = 1` and Padding `P = 0` 
  - Output of `4x4x1`

![Screen Shot 2022-01-18 at 14 15 28](https://user-images.githubusercontent.com/36196866/149985898-a242a74a-9807-492c-81bc-88ba421d1767.png)

- It is also possible to convolve to multiple filters (image taken from the course):

![Screen Shot 2022-01-18 at 14 22 26](https://user-images.githubusercontent.com/36196866/149986997-4451a3d9-5a53-4481-b140-74d58f412f65.png)

- When using multiple filters, the output matrix have the shape of `(n+2p-f)/s)+1 x ((n+2p-f)/s)+1 x n_c`, where `n_c` corresponds to the number of filters used


## One Layer of a Convolutional Network <a name="one_layer_conv_network"></a>


