
## vanilla_seq2seq

### learning_rate

The learning rate was set at 1e-3 as the same from previous HW. And the result was good. So I set the learning_rate as 1e-3

### rnnSz

The text book trys a rnnSz of 64. Thus, I tried 64 at first, it works well

### embedSz

To control the time, I set the embedSz as 30 at first. The program took about 25 mins to run, and the result was good. Thus, the embedSz was set as 30.

### Window Size

window size is the maximun sentence length.

## attn_seq2seq

The hyperparameters were set as the same with the vanilla_seq2seq. And they worked well.

### attention weight size

The attention weight has the size of [12,12], 12 is the maximum length for both english and french sentence.


```python

```
