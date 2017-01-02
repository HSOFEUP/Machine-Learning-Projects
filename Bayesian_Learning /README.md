This is a Matlab program that calculates the maximum likelihood estimation on the training set. Consider the prior defined as  
```P(C1|sigma) = 1.0/(1+exp(-sigma))``` and ```P(C2) = 1-P(C1)```  
Using the learned Bernoulli distributions and the given prior function, classify the samples in the validation set using your classification rules for   
```sigma =  5,  4,  3,  2,  1, 0, 1, 2, 3, 4, 5.```  
Finally, choose the best prior (the one that gives the lowest error rate on the validation set) and use it to classify the samples in the test set.  
  
```Bayes_Learning.m```: Train and learn the model. The function returns the outputs (p1: learned Bernoulli parameters of the first class, p2: learned Bernoulli parameters of the second class, pc1: best prior of the first class, pc2: best prior of the second class). It also prints a table of error rates of all priors.  
```Bayes_Testing.m```: Predict on the test set. The function prints the error rate on the test dataset.  
 ```Error rate```: Error rate is the percentage of wrongly classified data points divided by the total number of classified data points.
