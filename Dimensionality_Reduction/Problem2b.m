function Problem2b(traindata, testdata, k) %k is an array of [1 3 5 7]
%Run: Problem2b('optdigits_train.txt', 'optdigits_test.txt', [1 3 5 7]);
train_data = importdata(traindata);
test_data = importdata(testdata);
features = train_data(:,1:size(train_data,2)-1);
digit = train_data(:,size(train_data,2));
features_test = test_data(:,1:size(test_data,2)-1);
digit_test = test_data(:,size(test_data,2));
me = mean(features); % me is mean
centered = bsxfun(@minus, features, me); %features data is centered

%implementing pca for training data
covariancematrix=cov(features);
[V,D] = eig(covariancematrix);
D=diag(D);
[D, order] = sort(D, 'descend'); 
sorted_eigenval=V(:,order); %sorted eigenvalues with corresponding eigenvectors
finaldata=centered*sorted_eigenval(:,1:2); %projection; converting to 2 dimension

%implementing pca for test data
centered_test = bsxfun(@minus, features_test, me); %me remains the same
finaldata_test =centered_test*sorted_eigenval(:,1:2); %projection

%outputing the final training and test data to text file as the inputs for myKNN function
dlmwrite('A.txt',[finaldata digit]);
dlmwrite('B.txt',[finaldata_test digit_test]);
class = myKNN('A.txt','B.txt', k); %getting the trained labels for each k


%plotting for k={1,3,5}
for index = 1:3
c = class(:,index);
subplot(1,3,index);
projdata = [finaldata_test c]; %using finaldata_test
plot(finaldata_test(c==0,1),finaldata_test(c==0,2), 'ob'); hold on;
plot(finaldata_test(c==1,1),finaldata_test(c==1,2), '^r'); hold on;
plot(finaldata_test(c==2,1),finaldata_test(c==2,2), 'xc'); hold on;
plot(finaldata_test(c==3,1),finaldata_test(c==3,2), 'hg'); hold on;
plot(finaldata_test(c==4,1),finaldata_test(c==4,2), 'sk'); hold on;
plot(finaldata_test(c==5,1),finaldata_test(c==5,2), 'dy'); hold on;
plot(finaldata_test(c==6,1),finaldata_test(c==6,2), '*g'); hold on;
plot(finaldata_test(c==7,1),finaldata_test(c==7,2), '*k'); hold on;
plot(finaldata_test(c==8,1),finaldata_test(c==8,2), '<m'); hold on;
plot(finaldata_test(c==9,1),finaldata_test(c==9,2), '+r'); 
end