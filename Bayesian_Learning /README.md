This is a Matlab program that calculates the maximum likelihood estimation on the training set. Consider the prior defined as  
```P(C1|sigma) = 1.0/(1+exp(-sigma))``` and ```P(C2) = 1-P(C1)```  
Using the learned Bernoulli distributions and the given prior function, classify the samples in the validation set using your classification rules for   
```sigma =  5,  4,  3,  2,  1, 0, 1, 2, 3, 4, 5.```  
Finally, choose the best prior (the one that gives the lowest error rate on the validation set) and use it to classify the samples in the test set.  
  
```Bayes_Learning.m```: train and learn the model  
```Bayes_Testing.m```: predict on the test set
