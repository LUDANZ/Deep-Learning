
### Strides for tf.nn.conv2

To keep more features, the vertical and horizontal strides were set as 1. Thus, the strides for tf.nn.conv2 in the two convolutions were both set as [1,1,1,1]. Larger strides were also tried but the accuracy became less if the strides became larger.


### Ksize for tf.nn.max_pooling

With the same idea to keep more features. The Ksizes were set as 2 for both vertical and horizontal as first. Then, larger ksizes were tried but with less accuracy. Thus, the ksizes are all [1,2,2,1]

### How to decide the size of the hidden layer

I tried the size from 100 to 2000. And I found a plateau of accuracy after 500. Thus, the size of hidden layer was set as 512 as the power of 2.

#### Reference:

Eugene Charniak, Introduction to Deep Learning, 2017


```python

```
