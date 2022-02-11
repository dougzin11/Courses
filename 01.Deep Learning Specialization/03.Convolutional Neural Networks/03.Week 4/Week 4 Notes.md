## Table of contents
1. [Face Verification vs Face Recognition](#face_recognition)
2. [One Shot Learning](#one_shot_learning)
3. [Siamese Network](#siamese_network)
4. [Triples Loss](#triples_loss)
5. [Face Verification and Binary Classification](#face_verification_and_binary_classification)
6. [What is Neural Style Transfer](#what_is_neural_style_transfer)
7. [What are deep ConvNets learning?](#what_are_convenets_learning)
8. [Cost function](#cost_function)
9. [Content Cost Function](#content_cost_function)
10. [Style Cost Function](#style_cost_function)


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
## What is Neural Style Transfer <a name="what_is_neural_style_transfer"></a>
- Neural Style Transfer takes a Content Image (`C`) and a Style Image (`S`) and generates another image `G` based on the content image and style image. The picture below shows the idea behind the Neural Style Transfer **(image taken from the course)**

  <img width="978" alt="Screen Shot 2022-02-07 at 19 57 16" src="https://user-images.githubusercontent.com/36196866/152886276-35292c8c-8e69-4a97-a5e2-b7528e4ee932.png">

- In order to implement Neural Style Transfer, you need to look at the features extracted by ConvNet at various layers: the shallow and the deeper layers of a ConvNet


## What are deep ConvNets learning? <a name="what_are_convenets_learning"></a>
- Shallow layers: learn more simple representations like edges, horizontal lines, etc.
- Deeper layers: learn more complex representations like dogs, texts, etc.
- The image below shows how the complexity of the representations tent to get higher when we go deeper in the network **(image taken from the course)**

  <img width="944" alt="Screen Shot 2022-02-07 at 20 09 52" src="https://user-images.githubusercontent.com/36196866/152887619-d7114193-17ee-4bf4-b839-58617173d87a.png">


## Cost function <a name="cost_function"></a>
- Given a content image `C`, a style image `S` and a generated image `G`, we will define a cost function `J(G)` as follows:
  - `J(G) = alpha * J_content(C, G) + beta * J_style(S, G)`
    - J(C, G) measures how similar is the generated image to the Content image
    - J(S, G) measures how similar is the generated image to the Style image
    - `alpha` and `beta` are hyperparameters
- To find t he generated image `G`, here is what we do:
  1. Initiate `G` randomly
  2. Use Gradient Descent to minimize `J(G)`
    a. `G = G - dG`
- See below an example **(images taken from the course)**:
  - Goal is to generate an image combining the content of the image on the left with the style of the image on the right

    <img width="314" alt="Screen Shot 2022-02-07 at 20 21 36" src="https://user-images.githubusercontent.com/36196866/152888898-7d018515-d17f-42f6-b8f8-91409d12e556.png">

  - Following the above process, here is what we might end-up seeing:

    <img width="565" alt="Screen Shot 2022-02-07 at 20 22 05" src="https://user-images.githubusercontent.com/36196866/152888913-a9b792d8-0165-4a00-a5e8-428876256d58.png">


## Content Cost Function <a name="content_cost_function"></a>
- Pick a hidden layer `l`
  - Usually, `l` is not too shallow and not too deep (i.e.a layer in the middle of the network)
- Use pre-trained ConvNet
- Let `a[l]_c` and `a[l]_g` be the activation of the layer `l` on the images
- If `a[l]_c` and `a[l]_g` are similar, then they will have the same content
  - `J_content(C, G)[l] = 1/2 * || a[l]_c - a[l]_g ||**2` (element-wise difference between Content and Generated image)


## Style Cost Function <a name="style_cost_function"></a>
- Say you are picking a layer `l` to calculate/measure the style
- Style is defined as the covariance between activations across channels in the layer `l` - see image below **(image taken from the course)**

  ![Screen Shot 2022-02-11 at 11 10 30](https://user-images.githubusercontent.com/36196866/153606689-cb39811f-07fe-4a57-8ec8-1b4ac478335c.png)

- In the end, you want the covariance in the Style image channels to also appear in the Generated image channels
- Style matrix (Gram matrix):
  - Let `a[l][i, j, k]` be the activation at layer `l`, where `i` indexes into the `Height`, `j` into the `Width` and `k` into the `Channels`
  - Calculate `G[l]` - shape of `n_c[l] x n_c[l]` for both the Style image and Generated image **(image taken from the course)**
    - This calculates the covariance among channels

      <img width="465" alt="Screen Shot 2022-02-11 at 11 25 40" src="https://user-images.githubusercontent.com/36196866/153609025-dba6477d-cf12-47c9-9d71-1b12d47329af.png">

  - Finally, the Style Cost Function can be given by the element-wise difference between the 2 Gram matrixes:

     ![Screen Shot 2022-02-11 at 11 26 49](https://user-images.githubusercontent.com/36196866/153609324-d35579f8-87b9-4566-9522-ddcb972bd5cc.png)

- It turns out that you get more visually pleasing results if you use the style cost function from multiple different layers. So, the overall style cost function can be defines as:
  - `J(S, G) = sum(lambda[l] * J(S,G)[l])`, for all layers considered
    - `lambda[l]` is an additional hyperparameter
