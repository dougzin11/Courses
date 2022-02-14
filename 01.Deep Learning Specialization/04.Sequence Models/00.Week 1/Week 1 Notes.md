## Table of contents
1. [Notation](#notation)
2. [Recurrent Neural Network Model](#recurrent_neural_network_model)


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

  
## Recurrent Neural Network Model <a name="recurrent_neural_network_model"></a>
- 
  
