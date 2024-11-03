# Chapter 2 Working with the Text Data

The following figues shows the phases for the coding an LLM.


![coding llm](resources/stage1.svg) 

In this Chapter 2 we will be going through the step 1 of the Stage 1 of the Creating a LLM Application





# Understanding Terminologies 
##  Word Embedding: 
Word embeddings are essential for enabling neural networks to process textual data, as these models cannot interpret raw text directly. Embeddings create a mapping from discrete entities, such as words, into a continuous vector space, allowing semantic relationships to be represented numerically and making the data usable for machine learning models.

Understanding the concept of embedding spaces can be likened to a 2D Cartesian coordinate system, where points with similar values are positioned close to each other. In higher-dimensional embedding spaces (e.g., 768 dimensions), words that are semantically similar are represented by vectors that are close together, while dissimilar words are positioned farther apart. This method provides a structured way to represent linguistic relationships in a continuous vector space, enabling more effective processing of language by machine learning models.

### Word2Vec
Word2Vec Overview

Word2Vec is a prominent algorithm developed by a team led by Tomas Mikolov at Google in 2013, designed to generate word embeddings, which are dense vector representations of words. These embeddings capture semantic meanings and relationships between words in a continuous vector space, facilitating various natural language processing (NLP) tasks.

#### Algorithmic Structure

Word2Vec operates on the principle of predicting context words given a target word, employing either of two model architectures:

- Continuous Bag of Words (CBOW): In this approach, the model predicts the target word based on its surrounding context words. The context words are typically defined by a sliding window of a specified size around the target word. For example, given the context words “the,” “cat,” and “sat,” the model might predict the target word “on.”

- Skip-Gram: The reverse of CBOW, the Skip-Gram model predicts the context words from a given target word. This architecture is particularly effective when working with smaller datasets and allows the model to capture a wider range of contextual relationships.

#### Training Methodology

The training process of Word2Vec involves utilizing a neural network, which is typically shallow, consisting of one hidden layer. The algorithm adjusts the weights of the network through backpropagation based on the prediction errors. The training objective is to maximize the likelihood of predicting context words from the target words, which inherently optimizes the placement of words in the vector space.

#### Output and Application

The output of the Word2Vec model is a set of word vectors where words with similar meanings are located closer together in the vector space. These embeddings can be utilized in various applications, including but not limited to:

- Sentiment analysis
- Document classification
- Machine translation
- Information retrieval

By effectively capturing semantic relationships, Word2Vec has become a foundational technique in the field of NLP, enabling machines to understand and process human language more effectively.

## Tokenizing Text 
Before the embeddings of the wrods are generated the words should be tokenized. Tokens create the individual words from the sentences that is passed to it. 

### Simple Tokenizer 

```python 
class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.string_to_int = vocab  
        self.int_to_str = { i:s for s,i in vocab.items()}


    def encode(self,text): 
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.string_to_int[s] for s in preprocessed]
        return ids 

    def decode(self, ids): 
        text = " ".join([self.int_to_str[x] for x in ids]) 
        text = re.sub(r'([,.:;?_!"()\']|--|\s)', r'\1', text)
        return text 
        

```

### BPE Tokenizer 


LLMs are pretrained by predicting the next word in the sequence 


# Encoding Word Position 
LLMs are based on the self attention mechanism which doesn't take into consideration about the position or order of the token within the sequence. For reproducibility purpose we need to have some sorts of position dependent encoding of the token IDs. 

## Relative Positional Embedding 


## Absolute Position Embedding 


### References
- Raschka, Sebastian. *Build a Large Language Model (From Scratch)*. Manning Pubication, 2024. Available at: [link](https://www.manning.com/books/build-a-large-language-model-from-scratch)

