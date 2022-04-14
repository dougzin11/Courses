## Table of contents
1. [Word Representation](#word_representation)
2. [Using word embeddings](#using_word_embeddings)


# Introduction to Word Embeddings
## Word Representation <a name="word_representation"></a>
- So far, we've been representing words using a vocabulary of words and one-hot vectors **(image taken from the course)**:

  <img width="339" alt="Screen Shot 2022-04-14 at 11 13 39" src="https://user-images.githubusercontent.com/36196866/163409106-132d617e-aff4-47fe-8c60-49ff37263745.png">
  
  - `O_idx` corresponds to the word where the index `idx` is represented by a value of 1 in the one-hot vector
- One of the weaknesses of this representation is that it does not allow an algorithm to generalize across words:
  - Ex: `"I want a glass of orange _____"`. A model should predict the blank word as `juice`
  - Ex: `"I want a glass of apple _____"`. A model should also predict the blank word as `juice`
  - However, from the representation of the words above, the words `apple` and `juice` do not hold any relationship between them. So it is hard for the algorithm to generalize and know that the second example might also be `juice` if trained by seeing only the 1st sentence example
    - This is because the inner product between any of the one-hot encoding vectors is zero. In addition, the distance between any one-hot encoding vector is the same (i.e. the distance between `apple` and `orange` is the same as the distance between `apple` and `king`, for example)

- Therefore, we can rely on a different approach to represent words: featurized representation **(image taken from the course)**:

  <img width="957" alt="Screen Shot 2022-04-14 at 11 26 27" src="https://user-images.githubusercontent.com/36196866/163411429-8509ce0a-672e-4715-b221-630899fb9eda.png">

  - The above example shows 300 features used to represent a word. This way, each word will be represented by a 300-dimensional vector.
  - `e_idx` is the notation used to describe the word `O_idx` in the one-hot enconding representation
  - If we go back to the 2 example sentences above, we can see that now `apple` and `orange` hold similar features, making it easier for the algorithm to generalize
  - This type of representation is called **word embeddings**


## Using word embeddings <a name="using_word_embeddings"></a>


# Learning Word Embeddings: Word2Vec & Glove


# Applications Using Word Embeddings
