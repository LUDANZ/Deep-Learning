
### Embedding Size

The text book says that the embedding size of 100 is normal. Thus, I tried 100 at first, and the result was good. Thus, I use 100 for the embedding size.

### Size of Hidden Layer

If no hidden size was added, the program took about 14 minutes to run. To make sure the running time didn't increase too dramatically, I tried the hidden size of 100, 200 and 500. It turned out that when the size was 200, the running time was about 15 minutes and the perplexity was 30 less than without hidden layer. Thus, I added a hidden layer with size 200 to the network.
