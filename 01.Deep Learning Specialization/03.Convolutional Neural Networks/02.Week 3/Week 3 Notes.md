## Table of contents
1. [Object Localization](#object_localization)
2. [Landmark Detection](#landmark_detection)
3. [Object Detection](#object_detection)
4. [Convolutional Implementation of Sliding Windows](#convolutional_implementation_of_sliding_windows)
5. [Bounding Box Predictions](#bounding_box_predictions)
6. [Intersection Over Union](#intersection_over_union)
7. [Non-max Suppresion](#non_max_suppression)
8. [Anchor Boxes](#anchor_boxes)
9. [YOLO Algorithm](#yolo_algorithm)


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
- In order to produce more accurate bounding boxes, we can reccurr to a different algorithm: YOLO (you only look once)
- The idea behind YOLO is as follows:
  - Take the input image and place a grid on top of it (the more granular the grid, the more accurate the bounding box is). In the example below, we place a `3x3` grid **(image taken from the course)**

    ![Screen Shot 2022-01-31 at 18 49 12](https://user-images.githubusercontent.com/36196866/151878776-8e215955-f563-47a9-9e3a-2fec8ec26413.png)

    - For each grid, we will need to have labels for training the model. The labels for each grid will look like as below
      ```
      Y = [
          Pc, # corresponds to the probability that the midpoint of the object is in the grid
          b_x, # bounding box parameter
          b_y, # bounding box parameter
          b_h, # bounding box parameter
          b_w, # bounding box parameter
          c_1, # probability of the class 1
          c_2, # probability of the class 2
          ...
        ]
      ```
      - For cases where the object falls in 2 different grid cells, we assign the object to the grid cell that contains its midpoint. 
    - Since each grid cell should have an `8` dimensional output vector. Since we have a `3x3` grid, then the target output will have a size of `3x3x8`
  - The input image will pass through a Conv Net and have an output of the size `3x3x8`
    - The Conv Net will learn to map the input image to the output Y
  - As long as we don't have more than 1 object in the same grid cell, YOLO should work
    - We will see later how to address this problem
- How to specify the bounding box:
  - First, we need to set the origin for the grid (e.g. upper left corner is the origin `(0,0)`, which makes the bottom right corner assume the `(1,1)` value)
  - Set `b_x` and `b_y` accordingly (i.e. calculate the mid point of the object related to the origin) - `b_x` and `b_y` have to be between 0 and 1
  - `b_h` and `b_w` are specified as a fraction of the overall height and width of the grid cell - `b_h` and `b_w` can be greater than 1
- **Important notes**:
  - YOLO allows your network to output bounding boxes of any aspect ratio, as well as, output much more precise coordinates that aren't just dictated by the size of your sliding windows classifier
  - This is a convolutional implementation (you're not implementing this algorithm nine times on the `3x3` grid)


## Intersection Over Union <a name="intersection_over_union"></a>
- Intersection Over Union is a function used to evaluate your object detection algorithm
- Intersection Over Union (IoU) calculates the intersection over the predicted bounding box and the ground truth bounding box
- Take a look at an example below **(image taken from the course)**:
  
  ![Screen Shot 2022-01-31 at 19 51 19](https://user-images.githubusercontent.com/36196866/151886208-e622fb3a-ee2a-4163-b929-fd80abd72f1e.png)
  
  - Purple box: predicted output
  - Red box: ground truth output
  - `IoU = intersection area / union area`
- The higher `IoU` the better


## Non-max Suppresion <a name="non_max_suppression"></a>
- One of the problems of Object Detection is that your algorithm may find multiple detections of the same objects. Rather than detecting an object just once, it might detect it multiple times
- Non-max suppression is a way for you to make sure that your algorithm detects each object only once
- Non-max suppression algorithm (assuming we only have 1 output class):
  1. Discard all predicted bounding boxes that are below a specific threshold (e.g. `0.6`)
  2. While there are any remaining boxes:
    1. Pick the bounding box associated with the largest `P_c` output 
    2. Discard all the other boxes that have a hight overlap (i.e. high `IoU` greather than a specific threshold (e.g. `0.5`)) with the chosen box in the previous step
- If there are `c` classes/object you want to detect, you should run the Non-max suppression `c` times, once for every output class


## Anchor Boxes <a name="anchor_boxes"></a>
- One of the problems with object detection as you have seen it so far is that each of the grid cells can detect only one object. What if a grid cell wants to detect multiple objects? 
  - Imagine you have overlapping objects where their midpoint are almost in the same place and both of them fall into the same grid cell **(image taken from the course)**

    ![Screen Shot 2022-01-31 at 20 17 53](https://user-images.githubusercontent.com/36196866/151888867-b2601a97-4909-407b-af38-917963aae1d8.png)

  - Having an `Y` output of `[ Pc, b_x, b_y, b_h, b_w, c_1, c_2, c_3]` makes it impossible to detect 2 objects at the same time (we need to pick only 1). However, we can use the idea of anchor boxes
- You can use the idea of anchor boxes:
  - We are going to pre-defined different shapes called anchor boxes (e.g. 2 anchor boxes, 5 anchor boxes, etc.). Image below shows 2 anchor boxes **(image taken from the course)**

    ![Screen Shot 2022-01-31 at 20 22 27](https://user-images.githubusercontent.com/36196866/151889330-20624402-4094-4266-abe3-07c4289db369.png)

  - Now, we are going to associate the output `Y` to each anchor box:
    ```
       Y = [
          Pc, # Anchor box 1
          b_x, # Anchor box 1
          b_y, # Anchor box 1
          b_h, # Anchor box 1
          b_w, # Anchor box 1
          c_1, # Anchor box 1
          c_2, # Anchor box 1
          c_3, # Anchor box 1
          Pc, # Anchor box 2
          b_x, # Anchor box 2
          b_y, # Anchor box 2
          b_h, # Anchor box 2
          b_w, # Anchor box 2
          c_1, # Anchor box 2
          c_2, # Anchor box 2
          c_3 # Anchor box 2
        ]
    ```
  - Since the anchor box 1 is more similar to the shape of a pedestrian (in the example given above), then the first 8 elements of the output `Y` vector will be associated with detecting the presence of a pedestrian. The anchor box 2 (the remaining elements of `Y`) will then be associated with detecting the presence of a car
    - Similarity is measure in terms of `IoU`. The object is assigned to the anchor box with highets IoU
- To summarize:
  - Without anchor box: 
    - the object is assigned to the grid cell that contains object's midpoint (i.e. each object is assigned to a `grid cell`)
    - In our example, our output `Y` would have a shape of `3x3x8` (`3x3` comes from the fact we use a `3x3` grid and `8` comes from the fact we want to predict 3 classes)
  - With anchor boxes: 
    - the object is assigned to the grid cell that contains object's midpoint and also assigned to the anchor box with highest `IoU` (i.e. each object is assigned to the pair `grid cell, anchor box`)
    - In our example, our output `Y` would have a shape of `3x3x16` (`3x3` comes from the fact we use a `3x3` grid and `16` comes from the fact we want to predict 3 classes and we use 2 anchor boxes)
- Cases when anchor boxes does not perform well:
  - When there are 3 objects in the same grid cell but we use fewer anchor boxes
  - When there are objects with similar shape in the same grid cell


## YOLO Algorithm <a name="yolo_algorithm"></a>
- The above sections cover the idea behind the training phase of the YOLO algorithm
- In this section, we will focus what happens during prediction time:
  - If you are using `N` anchor boxes, each grid cell will output `N` bounding boxes (even if they have very low probabilities)
  - Get rid of the low probability predictions
  - We will then apply non-max suppression
  - See an example below where we are using `2` anchor boxes
    - Each grid cell will have 2 predicted bounding boxes

      ![Screen Shot 2022-02-01 at 09 28 18](https://user-images.githubusercontent.com/36196866/151968301-f2d8e34a-f93c-4833-b150-5d0a0eefefe0.png)
  
    - Get rid of low probability predictions

      ![Screen Shot 2022-02-01 at 09 27 52](https://user-images.githubusercontent.com/36196866/151968242-610c01cd-51f2-4476-86fd-7d9829d428ec.png)

    - For each class, use non-max suppression to generate final predictions

      ![Screen Shot 2022-02-01 at 09 29 49](https://user-images.githubusercontent.com/36196866/151968487-36c739c6-24dd-4ce7-b220-bd415dec8975.png)
  



