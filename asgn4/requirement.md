# Assignment 4: Trigram Language Model

## Goal:

We will be building a Trigram Language Model with Word Embeddings for language modeling the Penn Treebank Corpus.

## Data:

You can find the data for this task in the following location:

Training Data: /course/cs1470/asgn/trigram_lm/train.txt
Development Data: /course/cs1470/trigram_lm/dev.txt

## Steps:

Preprocess the Train and Development Data, building the vocabulary, tokenizing, etc.
Set up placeholders for inputs and outputs.
Initialize parameters for the model (however you like!)
Implement the forward pass for the model:
You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
You must return a distribution over the vocabulary (probability of the third word given the first two).
Calculate the loss for the model:
Calculate cross-entropy loss.
Set up the training step:
Use a learning rate of 1E-4
Use an Adam Optimizer [tf.train.AdamOptimizer]
Train your model (details below).
Print the final perplexity of the development set.

## Notes:

Train for 1 epoch, with a batch size of 20. You must abide by this - we will not grade any submission that uses an alternate batch size or learning rate.

We will be evaluating your language model on a held-out test set that you will not have access to during the development process. As such, it will be necessary to ensure your models do not over-fit to the development data.
