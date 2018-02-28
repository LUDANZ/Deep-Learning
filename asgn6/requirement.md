# Assignment 6: Sequence to Sequence Machine Translation

## Goal:
We will be building a simple set of models for performing Sequence-to-Sequence Machine Translation. Instead of building a full, complex machine translation pipeline, we will build a simple model that takes in short (< 12 word) French sentences, and outputs the English translations.

The first model we will build is a vanilla sequence-to-sequence model that uses an RNN encoder to encode French sentences, and a separate RNN Decoder to generate the translations. For this model, we will be passing the final hidden state of the Encoder as the initial state of the Decoder.

The second model we will build is a sequence-to-sequence model with "pseudo-attention." Instead of implementing the complex attention mechanisms discussed in class, we will instead be treating attention as a fixed table that, for each pair of English and French word positions, returns a corresponding weight.


## Vanilla Seq2Seq Steps:
Preprocess the Train and Development Data, building the different vocabularies, tokenizing, padding, etc.
Set up placeholders for inputs and outputs.
Initialize parameters for the model (however you like!)
Implement the forward pass for the model:
You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
Implement the Encoder-Decoder Seq2Seq Model as described in lecture. You are only allowed to use GRU cells for Encoder and Decoder RNNs.
Calculate the loss for the model:
Calculate sequence cross-entropy loss (tf.contrib.seq2seq.sequence_loss) - supplying a weight tensor to mask out all of the batch padding.
Set up the training step:
Use an Adam Optimizer [tf.train.AdamOptimizer]
Train your model (details below).
Print the per-symbol accuracy on the development set.
Psuedo-Attention Seq2Seq:
Follow the same steps as above, but initialize a variable of size (max_french_len, max_english_len). Use this variable in between your Encoder and Decoder in your Seq2Seq model to implement pseudo-attention, as described in lecture.

## Notes:
Train for 1 epoch, with a batch size of 20 You must abide by this - we will not grade any submission that uses an alternate batch size or number of training steps.
