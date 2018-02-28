
### Embedding Size

The text book says that the embedding size of 100 is normal. Thus, I tried 100 at first, and the perplexity of development set was about 170. Then, I increased the embedding size by 100 and find that the perplexity arrived at a plateau after the embedding size get 800. To balance the running time and perplexity, the embedding size was set as 800.

### Size of LSTM Layer

The hidden size was set as 500 at first and it resulted in good results and running time. Thus, the LSTM size was set as 500.