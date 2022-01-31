## Table of contents
1. [Object Localization](#object_localization)
2. [Landmark Detection](#landmark_detection)
3. [Object Detection](#object_detection)
4. [Convolutional Implementation of Sliding Windows](#convolutional_implementation_of_sliding_windows)
5. [Bounding Box Predictions](#bounding_box_predictions)


# Detection algorithms
## Object Localization <a name="object_localization"></a>
- Image classification: task involves classifying an image. Usually, the image has only 1 object to classify **(image taken from the course)**
  
  ![Screen Shot 2022-01-26 at 20 14 19](https://user-images.githubusercontent.com/36196866/151262867-25366d97-7195-4630-be9b-8f1acac656f2.png)

- Classification with localization: task involves classifying an image and identifying the class location in the image (e.g. draw a rectangle surrounding the class). Usually, the image has only 1 object to classify and localize **(image taken from the course)**

  ![Screen Shot 2022-01-26 at 20 14 53](https://user-images.githubusercontent.com/36196866/151262928-4164645b-56d7-4bf6-9c5b-8564be00ee4c.png)

- Detection: task involves detecting all objects in the image (that may belong to different classes) and identifying the location of each object. Usually, the image has only **multiple** objects to classify and localize **(image taken from the course)**

  ![Screen Shot 2022-01-26 at 20 17 13](https://user-images.githubusercontent.com/36196866/151263146-1a712865-a5c0-4ba1-b084-cb1f57ae728d.png)

- Image classification with localization:
  - In order to output a bounding box, we change the neural network to output 4 **more** numbers: `b_x`, `b_y`, `b_w` and `b_h`
  - These 4 numbers parametrize the bounding box of the detected object, where:
    - (`b_x`, `b_y`) corresponds to the center point of the bounding box
    - `b_h` corresponds to the height of the bounding box
    - `b_w` corresponds to the width of the bounding box
  - The image below ilustrates these 4 parameters, considering that the upper left corner is `(0,0)` and the lower right corner is `(1, 1)` **(image taken from the course)**
  
    ![Screen Shot 2022-01-26 at 20 24 28](https://user-images.githubusercontent.com/36196866/151263941-d0526a69-4f23-4582-94ce-1c650f8bc297.png)

  - The target label `Y` can be defined as follows:
  ```
  Y = [
        Pc, # corresponds to the probability that there is an object in the image
        b_x, # bounding box parameter
        b_y, # bounding box parameter
        b_h, # bounding box parameter
        b_w, # bounding box parameter
        c_1, # probability of the class 1
        c_2, # probability of the class 2
        ...
      ]
  ```
    - A couple of target label `Y` examples:
      - There is an object of the class `2`
        ```
        Y = [
              1,
              0,
              0,
              100,
              100,
              0,
              1,
              0
            ]
        ```
      - There is no object in the image
        ```
        Y = [
              0,
              ?, # we don't care with the output here (no object)
              ?, # we don't care with the output here (no object)
              ?, # we don't care with the output here (no object)
              ?, # we don't care with the output here (no object)
              ?, # we don't care with the output here (no object)
              ?, # we don't care with the output here (no object)
              ?  # we don't care with the output here (no object)
            ]
        ```


## Landmark Detection <a name="landmark_detection"></a>
- You can have a neural network to output `X` and `Y` coordinates of important points in an image that you want your network to recognize. These points are called **landmarks**
- For example, if you are working on a face recognition problem you might want some points on the face like corners of the eyes, corners of the mouth, corners of the nose, and so on
  - The target label `Y` would be similar to what was presented before:
    ```
    Y = [
          Pc, # corresponds to the probability that there is a face in the image
          l_1x, # x coordinate of the landmark 1
          l_1y, # y coordinate of the landmark 1
          l_2x, # x coordinate of the landmark 2
          l_2y, # y coordinate of the landmark 2
          ...
    ```
  - The landmarks need to be consistent across all the labeled training data. For example, if landmark 1 represents the left corner of the left eye, this needs to hold true for all the other labeled images


## Object Detection <a name="object_detection"></a>
- If you are building a car detection algorithm, for example, you can follow an approach called **Sliding Windows Detection**:
- For this approach, here is what we do:
  1. Train a ConvNet on cropped images of cars, as shown below **(image taken from the course)**
    
      ![Screen Shot 2022-01-27 at 18 47 21](https://user-images.githubusercontent.com/36196866/151449158-2a3549c8-7395-4dd2-afbf-03f1d40aee2d.png)
 
  2. Pick a window size
  3. Split your image into many windows of the size you picked in the previous step (every part of the image needs to be covered)
  4. Feed each window to the ConvNet trained on step 1
  5. Repeat step 2, 3 and 4 using a different window size (this step can be repeated as many times as you want - i.e. you can pick several window sizes)
  6. Store the windows that the ConvNet model classified as containing a car

- Disadvantages of the Sliding Windows Dectetion:
  - High computational cost
  

## Convolutional Implementation of Sliding Windows <a name="convolutional_implementation_of_sliding_windows"></a>
- First, let's understand how we can turn fully connected layers (FC) into convolutional layers. The image below shows us how **(image taken from the course)**

  ![Screen Shot 2022-01-31 at 09 54 44](https://user-images.githubusercontent.com/36196866/151797155-2e0730a9-3c9e-4510-8ec2-93e0ae1855c1.png)

  - The image above shows we can use 400 filters of `5x5x16` to turn the input of `5x5x16` into an output of `1x1x400`. This is how we can transform FCs into convolutional layers
- Convolution implementation of sliding windows:
  - Suppose that you trained a Conv Net in a `14x14x3` image, as shown below **(image taken from the course)**:
  
    ![Screen Shot 2022-01-31 at 09 58 39](https://user-images.githubusercontent.com/36196866/151797662-c3fd4deb-6e92-425b-a9f6-663370d6d567.png)

  - However, say we have now a `16x16x3` image that we want to apply the sliding windows. What we could do is feed the `16x16x3` image to the trained Conv Net
    - In the image below **(image taken from the course)**, we can break down the `2x2x4` output image as follows:
      - upper left-hand corner corresponds to the same result as running the Conv Net in the upper left-hand corner of the `16x16x3` input image (pixels painted as blue in the image)
      - upper right-hand corner corresponds to the same result as running the ConvNet in the upper right-hand corner of the `16x16x3` input image
      - The same applies to all the other regions (i.e. lower right-hand corner and lower left-hand corner)
- This implementation, however, also has its pros and cons:
  - pros: reduces the computational cost associated with the sliding windows
  - cons: the position of the bounding boxes surrounding the object will not be as accurate as the original sliding windows technique

 
 ## Bounding Box Predictions <a name="bounding_box_predictions"></a>



