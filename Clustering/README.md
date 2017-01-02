This project consists of the implementation of expectation–maximization (EM) algorithm to estimate a mixture of Gaussian distributions for image compression.
  
```EMG.m```:   
A function to estimate a mixture of k Gaussian distributions and run it on the image file “stadium.bmp”. 
Cluster the pixels into k = {4, 8, 12} clusters and plot the compressed images for each value of k. 
It also plots the expected complete log-likelihood function after each E-step and M-step of the EM algorithm as a single curve for each value of k.
