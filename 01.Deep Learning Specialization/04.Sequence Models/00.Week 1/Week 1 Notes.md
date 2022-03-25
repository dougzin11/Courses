## Table of contents
1. [Notation](#notation)
2. [Recurrent Neural Network Model](#recurrent_neural_network_model)
3. [Backpropagation through time](#backpropagation_through_time)
4. [Different types of RNN](#different_types_of_rnn)
5. [Language model and sequence generation](#language_model_and_sequence_generation)
6. [Vanishing gradients with RNNs](#vanishing_gradients)
7. [Gated Recurrent Unit](#gated_recurrent_unit)
8. [Long Short Term Memory (LSTM)](#lstm)
9. [Bidirectional RNN](#bidirectional_rnn)


# Recurrent Neural Networks
## Notation <a name="notation"></a>
- Motivating example: Named entity recognition example:
    - Both elements has a shape of 9. 1 means its a name, while 0 means its not a name 
    - X: "Harry Potter and Hermoine Granger invented a new spell."
    - Y: 1 1 0 1 1 0 0 0 0
- Notation:
  - The first element of `x` is denoted by <code>x<sup><1></sup></code>, the second <code>x<sup><2></sup></code> and so on
    - <code>x<sup><1></sup></code> = Harry
    - <code>x<sup><2></sup></code> = Potter
  - Similarly, the first element of `y` is denoted by <code>y<sup><1></sup></code>, the second <code>y<sup><2></sup></code> and so on
    - <code>y<sup><1></sup></code> = 1
    - <code>y<sup><2></sup></code> = 1
  - <code>T<sub>x</sub></code> is the size of the input sequence and <code>T<sub>y</sub></code> is the size of the output sequence
    - <code>T<sub>x</sub></code> = <code>T<sub>y</sub></code> = 9 in the motivating example
  - <code>x<sup>(i)<t></sup></code> is the element `t` of the sequence of input vector `i` (same thing applies to <code>y<sup>(i)<t></sup></code>)
  - The length <code>T<sub>x</sub></code> and <code>T<sub>y</sub></code> can be different across the examples. 
    - The <code>T<sub>x</sub><sup>(i)</sup></code> denotes the size of the input sequence of the training example `i`
- To represent words:
  - We need a vocabulary list that contains all the words in our target set
  - Create a one-hot encoding sequence for each word in the dataset given the vocabulary picked in the previous step
    - We can add a token in the vocabulary with the name `<UNK>` (which stands for unknown text) and recur to this token when encountering words that are not in the vocabulary
  - See below an example **(image taken from the course)**:
    
    <img width="662" alt="Screen Shot 2022-02-16 at 19 11 57" src="https://user-images.githubusercontent.com/36196866/154366085-21403380-9530-47cc-9f57-170c65dc45dc.png">
  
## Recurrent Neural Network Model <a name="recurrent_neural_network_model"></a>
- There are 2 problems when using a standard neural network for sequence tasks:
  1. Inputs and outputs can have different lengths in different examples
  2. Features learned across different positions of text/sequence are not shared
        - Similarly to CNNs, sharing features allows to significantly reduce the number of parameters in the model
- Recurrent Neural Networks don't have neither of the two problems mentioned above
- Example of Recurrent Neural Network:
  
    ![Screen Shot 2022-02-16 at 20 02 12](https://user-images.githubusercontent.com/36196866/154372093-9c802cb0-20c5-4069-923c-30c32e2e13df.png)

    - In the above example, <code>T<sub>x</sub></code> = <code>T<sub>y</sub></code>. The structure would change a little bit if they were not equal
    - <code>a<sup><0></sup></code> is usually initialized with zeros (can also have random values in some cases)
    - There are 3 parameters: 
        - <code>W<sub>ax</sub></code> : governs the connection of the input `X` to the hidden layers
        - <code>W<sub>aa</sub></code> : governs the horizontal connections
        - <code>W<sub>ya</sub></code> : governs the output predictions
    - This network does not use information coming from words later in the sequence:
        - "He said, Teddy Roosevelt was a great president"
        - "He said, Teddy bears are on sale"
        - The first 3 words are the same, but in the first example "Teddy" should be marked as a name and it would be useful to know the words later in the sequence. For this, we will later discuss Bidirectional RNN (BRNN)
    
- Forward propagation equations:
    - <code>a<sup><1></sup> = g<sub>1</sub>(W<sub>aa</sub>a<sup><0></sup> + W<sub>ax</sub>x<sup><1></sup> + b<sub>a</sub>)</code>
    - <code>y<sup><1></sup> = g<sub>2</sub>(W<sub>ya</sub>a<sup><1></sup> + b<sub>y</sub>)</code>
    - Generalizing:
        - <code>a<sup>< t ></sup> = g(W<sub>aa</sub>a<sup>< t-1 ></sup> + W<sub>ax</sub>x<sup>< t ></sup> + b<sub>a</sub>)</code>
        - <code>y<sup>< t ></sup> = g(W<sub>ya</sub>a<sup>< t ></sup> + b<sub>y</sub>)</code>
- We can simplify the general equations as follows:
    - <code>a<sup>< t ></sup> = g(W<sub>aa</sub>a<sup>< t-1 ></sup> + W<sub>ax</sub>x<sup>< t ></sup> + b<sub>a</sub>)</code>
        - <code>a<sup>< t ></sup> = g(W<sub>a</sub>[a<sup>< t - 1></sup>, x<sup>< t ></sup>] + b<sub>a</sub>)</code>, where:
            - <code>W<sub>a</sub> = [W<sub>aa</sub>, W<sub>ax</sub>]</code> (i.e. <code>W<sub>aa</sub></code> and <code>W<sub>ax</sub></code> stacked horizontally)
            - <code>[a<sup>< t - 1></sup>, x<sup>< t ></sup>] = [a<sup>< t - 1 ></sup> / x<sup>< t ></sup>]</code> (i.e. <code>a<sup>< t - 1 ></sup></code> and <code>x<sup>< t ></sup></code> stacked vertically)
    - <code>y<sup>< t ></sup> = g(W<sub>ya</sub>a<sup>< t ></sup> + b<sub>y</sub>)</code>
        - <code>y<sup>< t ></sup> = g(W<sub>y</sub>a<sup>< t ></sup> + b<sub>y</sub>)</code>


## Backpropagation through time <a name="backpropagation_through_time"></a>
- First, let's take a look at the Forward propagation step **(image taken from the course)**:
   
    ![Screen Shot 2022-03-07 at 19 54 22](https://user-images.githubusercontent.com/36196866/157131817-d28020c2-45d5-424b-a4a2-4b7cb61d4a7c.png)

    - Where <code>W<sub>a</sub></code>, <code>b<sub>a</sub></code>, <code>W<sub>y</sub></code>, <code>b<sub>y</sub></code> are shared across all the network
    
- In order to calculate the backpropagation step, we need to define a loss function. In the below case, we define the cross-entropy loss **(image taken from the course)**:
    
    <img width="540" alt="Screen Shot 2022-03-07 at 19 57 58" src="https://user-images.githubusercontent.com/36196866/157132276-14be51b4-79c1-478d-89e7-c24fc9247916.png">

    - The first equation is the loss for one example
    - The second equation is the loss for the whole sequence that is given (from `t` = 1 till <code>T<sub>y</sub></code>)
    - When calculating the backpropagation step, we calculate the derivatives and the information "flow" in the opposite direction as the forward propagation step. Therefore, we have a flow comming from <code>a<sup><T<sub>y</sub>></sup></code> in the direction of <code>a<sup><1></sup></code>. This is the reason why this backpropagation is also called backpropagation through time
    

## Different types of RNN <a name="different_types_of_rnn"></a>
- The previous example is a case of an architecture called **Many to Many** where the output and input have the same length
- In sentiment analysis problems, `X` is a text while `Y` is an integer (e.g. ranging from 1 to 5). This RNN architecture is called **Many to One** **(image taken from the course)**
    
    ![Screen Shot 2022-03-07 at 20 12 12](https://user-images.githubusercontent.com/36196866/157133866-3f5ffc46-da2a-414e-bb6a-3d3ed3844fd9.png)

- There is also a RNN architecture called **One to Many** that can appear when dealing with music generation problems, for example **(image taken from the course)**
    
    ![Screen Shot 2022-03-07 at 20 13 30](https://user-images.githubusercontent.com/36196866/157134032-f8a6f090-020f-4cfe-9575-5bf6282d9fef.png)

- We can also encounter **Many to Many** architectures where the input and output have different lengths. Machine translation is one area where it is common to find this type of architecture **(image taken from the course)**
    
    ![Screen Shot 2022-03-07 at 20 16 05](https://user-images.githubusercontent.com/36196866/157134290-91b4ac20-4f36-4cbf-8c17-fb75cd23e0d9.png)

- Summary of RNNs architecture **(image taken from the course)**
    
    <img width="715" alt="Screen Shot 2022-03-07 at 20 17 16" src="https://user-images.githubusercontent.com/36196866/157134446-40e712a1-adb6-4651-b89e-a8fb27c70bea.png">


## Language model and sequence generation <a name="language_model_and_sequence_generation"></a>
- Language model
    - Let's say we are solving a speech recognition problem and someone says a sentence that can be interpreted into two sentences:
        - The apple and pair salad
        - The apple and pear salad
    - Pair and pear sound the same, so how would a speech recognition application choose from the two
    - That's where the language model comes in. It gives a probability for the two sentences and the application decides the best based on this probability
- How to build language model with RNNs:
    - You need a large corpus of english text
    - Tokenize each sentence in your training set
    - You can add the `<EOS>` token to explicitly capture when sentences end
    - You can add the `<UNK>` token for unknown words

    
## Vanishing gradients with RNNs <a name="vanishing_gradients"></a>
- Suppose we have the 2 example sentences below:
    - "The cat, which already ate ..., was full"
    - "The cats, which already ate ..., were full"
    - This is one example of when language can have very long-term dependencies, where words much earlier can affect what needs to come much later in the sentence (was vs were)
- The basic RNN we have seen so far is not very good at capturing very long-term dependencies
    - The outputs are mainly influenced by the inputs that are closer
        - E.g. <code>y<sup><10></sup></code> is more influenced by <code>x<sup><9></sup></code> input than <code>x<sup><1></sup></code>
    - As we have discussed in Deep neural networks, deeper networks are getting into the vanishing gradient problem. That also happens with RNNs with a long sequence size
    - For computing the word "was", we need to compute the gradient for everything behind. Multiplying fractions tends to vanish the gradient, while the multiplication of large numbers tends to explode it
        - Weight may not be properly updated
- RNNs can also suffer from exploding gradients (although it is less common than vanishing gradients):
    - One solution to that is to recur to gradient clipping: if your gradient is more than some threshold - re-scale some of your gradient vector so that is not too big. So there are clipped according to some maximum value
    
    
## Gated Recurrent Unit <a name="gated_recurrent_unit"></a>
- The Gated Recurrent Unit is a modification to the RNN hidden layer that makes it much better capturing long-range connections and helps a lot with the vanishing gradient problems
- The basic RNN unit can be shown as the below diagram **(image taken from the course)**
    
    ![Screen Shot 2022-03-23 at 18 05 18](https://user-images.githubusercontent.com/36196866/159795553-25c488bf-e84f-49f4-a8a3-1aad3af87ba6.png)

- The simplified version of GRU is governed by the following equations **(image taken from the course)**
    
    ![Screen Shot 2022-03-23 at 18 22 34](https://user-images.githubusercontent.com/36196866/159797979-b1180791-df07-427b-92c8-edf16f003e47.png)
    
    - The `C` stands for memory cell and it holds the same value as `a`: <code>C<sup>< t ></sup></code> = <code>a<sup>< t ></sup></code>
    - The `C_tilda` calculates a candidate to possibly update `C`
    - The gate is going to decide whether or not we replace `C` by `C_tilda`
        - The update gate always hold values between 0 and 1
    - The simplified version of GRU can be drawn as follows **(image taken from the course)**
    
        ![Screen Shot 2022-03-23 at 19 00 13](https://user-images.githubusercontent.com/36196866/159803132-560a7f09-ea89-4e6d-a9f4-64df9d9cd21f.png)

- GRUs are good at solving the vanishing gradient problem since:
    - The update gate can easily assume values close to zero which, in turn, makes <code>C<sup>< t ></sup></code> very close to <code>C<sup>< t - 1 ></sup></code>
    
- The full version of GRU is governed by the following equations **(image taken from the course)**:
    
    ![Screen Shot 2022-03-23 at 19 16 12](https://user-images.githubusercontent.com/36196866/159805191-0b0d68bc-82d5-4767-8e1f-b59aa040fd29.png)

    - The full GRU contains a new gate that is used with to calculate the candidate `C`. The gate tells you how relevant is <code>C<sup>< t - 1 ></sup></code> to computing the next candidate for <code>C<sup>< t ></sup></code>

    
## Long Short Term Memory (LSTM) <a name="lstm"></a>
- In the LSTM, <code>C<sup>< t ></sup></code> != <code>a<sup>< t ></sup></code>
- The equations governing LSTM can be shown as follows **(image taken from the course)**:
    
    ![Screen Shot 2022-03-25 at 14 56 41](https://user-images.githubusercontent.com/36196866/160175886-6d427baf-7350-450c-892d-880a45783ffd.png)

    - We can notice that in LSTM we have:
        - Update gate
        - Forget gate
        - Output gate
- The LSTM can be drawn as follows **(image taken from the course)**:
    
    ![Screen Shot 2022-03-25 at 15 03 29](https://user-images.githubusercontent.com/36196866/160176768-26d511c6-df34-419a-8901-8e88720b58dd.png)

    
## Bidirectional RNN <a name="bidirectional_rnn"></a>
- The Bidirectional RNN structure can be represented as follows **(image taken from the course)**:
    
    ![Screen Shot 2022-03-25 at 15 20 44](https://user-images.githubusercontent.com/36196866/160179148-ae3f88ec-e6af-442d-b9ee-ccff55a2bc57.png)

- Part of the forward propagation goes from left to right, and part - from right to left. It learns from both sides
- To make predictions we use <code>ŷ<sup>< t ></sup></code> by using the two activations that come from left and right
    - <code>ŷ<sup>< t > </sup> = g(W<sub>y</sub>[a_forward<sup>< t ></sup>, a_backward<sup>< t ></sup>] + b<sub>y</sub>)</code>
- The blocks here can be any RNN block including the basic RNNs, LSTMs, or GRUs
    
    
## Deep RNNs <a name="deep_rnns"></a>
- In a lot of cases the standard one layer RNNs will solve your problem. But in some problems its useful to stack some RNN layers to make a deeper network
- For example, a deep RNN with 3 layers would look like this:
    
    ![Screen Shot 2022-03-25 at 15 45 16](https://user-images.githubusercontent.com/36196866/160182687-145a9185-9deb-43a7-b5f4-8d2d4d08547e.png)

    - Example of how you calculate `a`:
    
        <code>a<sup>[2]<3></sup> = g(W<sub>a</sub><sup>[2]</sup>[a<sup>[2] <2></sup>, a<sup>[1] <3></sup>] + b<sub>a</sub><sup>[3]</sup>)</code>
    - The RNN blocks don't just have to be standard RNN. They can also be GRU blocks or LSTM blocks, for example
