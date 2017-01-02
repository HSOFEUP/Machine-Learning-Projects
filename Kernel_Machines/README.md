### Given problem:

Implement your GD kernel perceptron using an RBF kernel and train it on non-linearly separable data, ```data3```. All the data is contained in matrix data3 and the corresponding labels are in vector theclass. 
Test your implementation of the GD kernel perceptron algorithm on the above data and plot the decision boundary and report the error rate.  
  
Apply the ```trainsvm``` function in Matlab using RBF kernel to the same data and plot the decision boundary obtained by the SVM 
in the same figure with the decision boundary of your GD kernel perceptron. Compare the results with your GD kernel perceptron implementation. 
Play with the ```boxconstraint``` parameter and explain how it changes the margin obtained (See below for the explanation).  
##### Explanation:
In the non­separable case, the penalty factor, C is the box constraint. When C increases, the weight of misclassifications increases too. 
This causes a greater deviation from the margin and leads to a stricter separation.
  
Once you have tested that your GD kernel perceptron works, train and evaluate
your implementation using the given subset of the optdigits dataset. The first training and test datasets digits49_train.txt and digits49_test.txt consists of
only digits 4 and 9. The second training and test datasets digits79_train.txt and digits79_test.txt consists of only digits 7 and 9. Report the training
and test error rates on both datasets.  
