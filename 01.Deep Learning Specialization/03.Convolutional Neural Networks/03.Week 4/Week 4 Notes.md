## Table of contents
1. [Face Verification vs Face Recognition](#face_recognition)
2. [One Shot Learning](#one_shot_learning)
3. [Siamese Network](#siamese_network)
4. [Triples Loss](#triples_loss)
5. [Face Verification and Binary Classification](#face_verification_and_binary_classification)


# Face Recognition
## Face Verification vs Face Recognition <a name="face_recognition"></a>
- Face Verification:
  - Input: image, name/ID
  - Output: whether the input image is that of the claimed person **(1-1 problem)
- Face Recognition:
  - Given a database of K persons and an input image
  - Output ID if the image is any of the K persons (or output "not recognized")
- Face Recognition is much harder than the Face Verification problem


## One Shot Learning <a name="one_shot_learning"></a>
- One of the challenges of face recognition is that you need to solve the one-shot learning problem: for most face recognition applications you need to be able to recognize a person given just one single image, or given just one example of that person's face
- Historically, deep learning does not work well with a small number of data. To solve the one-shot learning problem, we will learn a **similarity function**:
  - `d(image1, image2) = degree of difference between images`
  - If the difference is below some threshold (here called **Tau** `T`):
    - If `d(image1, image2) <= T`, then the faces are the same
    - If `d(image1, image2) > T`, then the faces are from different persons


## Siamese Network <a name="siamese_network"></a>
- A good way to calculate the similarity function is to use a Siamese Network
  - Given 2 images `x(1)` and `x(2)`, we pass both of these images in a Conv Network
  - Instead of taking the output after applying a Softmax function, we instead retrieve an encoding vector for each image (`f(x(1))` and `f(x(2))` are the encodings for the image `x(1)` and `x(2)`, respectively) - The example below shows an encoding vector of 128 **(image taken from the course)**

    ![Screen Shot 2022-02-01 at 18 51 39](https://user-images.githubusercontent.com/36196866/152058045-fece2943-7567-4667-aa28-900e61bc0b40.png)

  - We now compare the 2 encodings: `d(x(1), x(2)) = ||f(x(1)) - f(x(2))||**2`
  - The Network needs to learn parameters so that:
    - If `x(i)` and `x(j)` are the same person, then `d(x(i), x(j)) = ||f(x(i)) - f(x(j))||**2` should be small
    - If `x(i)` and `x(j)` are the different persons, then `d(x(i), x(j)) = ||f(x(i)) - f(x(j))||**2` should be large


## Triples Loss <a name="triples_loss"></a>
- In the terminology of the triplet loss, we will always compare 3 images at a time:
  - Anchor (A): image used as a reference
  - Positive (P): image that represents the same person as the anchor - you want the distance to be small when comparing to the anchor image
  - Negative (N): image that represents a different person - you want the distance to be big when comparing to the anchor image
- So formally, we want that:
  - Similarity between the anchor and positive to be greater than the similarity between the anchor and the negative:
    ```
    ||f(A) - f(P)||**2 <= ||f(A) - f(N)||**2
    ||f(A) - f(P)||**2 - ||f(A) - f(N)||**2 <= 0
    
    # To avoid allowing the Network to always predict 0
    # e.g. f(A) = f(P) = f(N) = 0 or f(A) = f(P) = f(N) = constant for all images
    # we make some changes to the formula above as follows:
    
    ||f(A) - f(P)||**2 - ||f(A) - f(N)||**2 <= -alpha #alpha is called a margin
    ||f(A) - f(P)||**2 - ||f(A) - f(N)||**2 + alpha <= 0 
    ```
- Triplet loss function: Given 3 images A, N, P:
  - `L(A, P, N) = max(||f(A) - f(P)||**2 - ||f(A) - f(N)||**2 + alpha , 0)`
    - Since we want `||f(A) - f(P)||**2 - ||f(A) - f(N)||**2 + alpha <= 0`, this means that as long the Triple Loss function above selects 0, we reached our objective. When the loss function returns a positive number, this then means we haven't achieved our objective
- Overall Cost Function:
  - `J = sum(L(A[i], P[i], N[i]))` for all triplets of images
- **Note**: You need multiple pictures of the same person (because you need some pairs of A and P for the same person)
- Choosing the triplets A, P and N:
  - During training if A, P, N are chosen randomly (subjected to A and P being the same person and A and N being different persons) then one of the problems is that the the loss function is easily satisfied
    - We may encounter A and N being totally different people, which makes things way too easy for the Network to learn
  - So what we want to do is choose triplets that are **hard** to train on
    - Cases where `d(A, P)` and `d(A, N)` are quite close: the learning algorithm has to try extra hard to have a margin of alpha between the the 2 distances


## Face Verification and Binary Classification <a name="face_verification_and_binary_classification"></a>
- Triplet loss is one way to learn the parameters of a Conv Net for face recognition. However, there's another way to learn these parameters as a straight binary classification problem
- Posing Face Verification problem as a Binary classification problem:
  - We can take the encoding vectors of the 2 images and input them to a logistic regression unit to then just make a prediction
  - The target output will be 1 if both of the images represent the same persons, and 0 if they represent different persons. The image below summarizes the process **(image taken from the course)**

    ![Screen Shot 2022-02-01 at 18 51 39](https://user-images.githubusercontent.com/36196866/152071612-cabc8d8d-fe12-4372-bfab-b770354c4ab7.png)

  - The final layer can be a Sigmoid function applied to the difference of the 2 encodings: `y = sigmoid(w_k * |sum(f(x(i))_k - f(x(j))_k))| + b`, where:
    - `k` represents the `k` component of the encoding vector (e.g. if the encoding vector has a dimension of `128`, then we would calculate the difference for each one of the `128` components)
    - `||` takes the absolute value of the difference between the 2 encoding vectors (we can use other distances like Euclidean distance)
    - `w` and `b` are parameters learned during training (like in traditional neural networks)

      ![Screen Shot 2022-02-01 at 21 10 39](https://user-images.githubusercontent.com/36196866/152072432-d7bafa69-b21e-4fa6-9de6-5fd736819f93.png)


# Neural Style Transfer
