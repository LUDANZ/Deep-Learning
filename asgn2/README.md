
### How to select **Learning Rate**

At first, the learning rate was set as 0.5. The interation times was set as 2000 and batch size was 200. The losses in the training interations were as following:

```
loss after 200 is: 0.358017414138
loss after 400 is: 0.223004448432
loss after 600 is: 0.168237949942
loss after 800 is: 0.137562428729
loss after 1000 is: 0.11687450585
loss after 1200 is: 0.102244946649
loss after 1400 is: 0.0910043836718
loss after 1600 is: 0.082075540307
loss after 1800 is: 0.0748234542785
loss after 2000 is: 0.0687420164007
Test Accuracy is: 97.78 %
```

We can observe a decrease in the loss. Thus, we can increase the learning rate. When the learning rate was increased to 1. The loss after 2000 interations became bigger than that of 0.5 learning rate. This means that 1 is too large. Then 0.8 was tried. And it turned out to have smaller loss and higher accuracy (about 97.95%). Finally, 0.8 was chosen as the learning rate. 

### How the batch size selected

At first, the batch size was set as 200 randomly and the running time was 50 seconds. When the batch size increased, the running time increased significantly. However, the accuracy didn't change too much (less then 0.1%). Thus, I chose not to increase the batch size and kept it as 200.

### How the interation time selected

The interation was 2000 times as first. The loss kept decreasing in the 2000 iteration. Thus, I tried to increase the interation times to 3000 which increased the running time to 70 seconds and the accuracy from about 97.95% to 98.05%. If the interation time increase to 4000, the running time would increase a lot. To balance the accuracy and the running time. I chose 2000 as the interation time.  

### How to decide the size of the hidden layer

I tried the size from 100 to 1200. And I found a plateau of accuracy after 1000. Thus, the size of hidden layer was set as 1000.

#### Reference:

Eugene Charniak, Introduction to Deep Learning, 2017
