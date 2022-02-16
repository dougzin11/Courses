## Table of contents
1. [Notation](#notation)
2. [Recurrent Neural Network Model](#recurrent_neural_network_model)
3. [Backpropagation through time](#backpropagation_through_time)


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
- 
    
    
