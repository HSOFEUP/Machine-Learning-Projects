function Problem2c(traindata, testdata, k) %k is an array of [1 3 5 7]
%Run: Problem2c('optdigits_train.txt', 'optdigits_test.txt', [1 3 5 7]);
train_data = importdata('optdigits_train.txt');
test_data = importdata('optdigits_test.txt');
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

cols = size(features,2);
%initialize determinant (de) to zero in order to determine if the inverse of Sw exists
de = 0; 
labels = unique(digit);

%find the D dimensions that the inverse of Sw exist
while de == 0  
    cols = cols - 1;
    finaldata=centered*sorted_eigenval(:,1:cols);
    %implementing LDA
    for i=0:9
        Xi = finaldata((digit==i),:);
        me = mean(finaldata); %me changes at every iteration 
        me_i = mean(Xi);
        centered_i = bsxfun(@minus, Xi, me_i);
        dim = size(finaldata,2);
        Sw = zeros(dim,dim);
        Sb = zeros(dim,dim);
        Sw = Sw + (centered_i' * centered_i); %Sw is within-class scatter matrix
        n = size(Xi,1);
        m = me_i - me;
        Sb = Sb + (n * (m' * m)); %Sb is between-class scatter matrix
    end
    de = det(Sw);
end

%initializing the Sw and Sb
Sw = zeros(cols,cols);
Sb = zeros(cols,cols);
for i=0:9
        Xi = finaldata((digit==i),:);
        me = mean(finaldata); %me changes at every iteration 
        me_i = mean(Xi);
        centered_i = bsxfun(@minus, Xi, me_i);
        Sw = Sw + (centered_i' * centered_i); %Sw is within-class scatter matrix
        n = size(Xi,1);
        m = me_i - me;
        Sb = Sb + (n * (m' * m)); %Sb is between-class scatter matrix
end

%From the value of cols, now we know the computed projection data has 61 dimensions.
%projecting using Sw and Sb (LDA)
[vec, lambda]=eig(inv(Sw)*Sb);
lambda = diag(lambda); 
[lambda, SortOrder]=sort(lambda,'descend');
sorted_vec = vec(:,SortOrder);
finaldata_lda = finaldata*sorted_vec(:,1:2); %projection %not sure need to center the data or not

%implementing PCA and LDA on test data

%PCA for test data
centered_test = bsxfun(@minus, features_test, mean(features)); %me remains the same
finaldata_test =centered_test*sorted_eigenval(:,1:cols); 

%LDA for test data
finaldata_lda_test = finaldata_test*sorted_vec(:,1:2); 

%outputing the final training and test data to text file as the inputs for myKNN function
dlmwrite('A2.txt',[finaldata_lda digit]);
dlmwrite('B2.txt',[finaldata_lda_test digit_test]);
class = myKNN('A2.txt','B2.txt', k);

%plotting for k={1,3,5}
for index = 1:3
c = class(:,index);
subplot(1,3,index);
projdata = [finaldata_lda_test c]; %using finaldata_test
plot(finaldata_lda_test(c==0,1),finaldata_lda_test(c==0,2), 'ob'); hold on;
plot(finaldata_lda_test(c==1,1),finaldata_lda_test(c==1,2), '^r'); hold on;
plot(finaldata_lda_test(c==2,1),finaldata_lda_test(c==2,2), 'xc'); hold on;
plot(finaldata_lda_test(c==3,1),finaldata_lda_test(c==3,2), 'hg'); hold on;
plot(finaldata_lda_test(c==4,1),finaldata_lda_test(c==4,2), 'sk'); hold on;
plot(finaldata_lda_test(c==5,1),finaldata_lda_test(c==5,2), 'dy'); hold on;
plot(finaldata_lda_test(c==6,1),finaldata_lda_test(c==6,2), '*g'); hold on;
plot(finaldata_lda_test(c==7,1),finaldata_lda_test(c==7,2), '>k'); hold on;
plot(finaldata_lda_test(c==8,1),finaldata_lda_test(c==8,2), '<m'); hold on;
plot(finaldata_lda_test(c==9,1),finaldata_lda_test(c==9,2), '+r');
end
disp(cols);