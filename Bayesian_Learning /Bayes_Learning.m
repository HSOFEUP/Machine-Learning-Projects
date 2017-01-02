function [p1, p2, pc1, pc2] = Bayes_Learning(training_data, validation_data);
%Run: [p1, p2, pc1, pc2] = Bayes_Learning('SPECT_train.txt','SPECT_valid.txt');

train_data = importdata(training_data);
valid_data = importdata(validation_data);  
train_cols = size(train_data,2);
class1 = train_data(train_data(:,train_cols)==1,:);
class2 = train_data(train_data(:,train_cols)==2,:);
class1_size = size(class1, 1);
class2_size = size(class2, 1);

%the following 2 loops (2 classes) calculate maximum likelihood estimation
p0_C1 = zeros(train_cols-1,1); %p(x=0|C1)
for i=1:train_cols-1
    p0_C1(i) = size(class1(class1(:,i)==0,i),1) / class1_size;
    %p0_C1(i)=(class1_size - sum(class1(:,i)))/class1_size;
    if p0_C1(i) == 1
        p0_C1(i) = 0.99999;
    end
end
p1_C1 = 1 - p0_C1;  %p(x=1|C1)

p0_C2 = zeros(train_cols-1, 1);  %p(x=0|C2)
for i=1:train_cols-1
    p0_C2(i) = size(class2(class2(:,i)==0,i),1) / class2_size;
    %p0_C2(i)=(class2_size - sum(class2(:,i)))/class2_size;
    if p0_C2(i) == 1
        p0_C2(i) = 0.99999;
    end
end
p1_C2 = 1 - p0_C2;  %p(x=1|C2)

%the following calculate posteriors by using discriminant function (5.30) in textbook
valid_size = size(valid_data, 1);
min=1;
best_prior_C1=0;
for sigma=-5:1:5
    PC1=1.0/(1+exp(-sigma));
    PC2=1-PC1;
    error = 0;
    for j=1:valid_size
        PC1X = (1-valid_data(j,1:22))*log(p0_C1) + valid_data(j,1:22)*log(p1_C1) + log(PC1);
        PC2X = (1-valid_data(j,1:22))*log(p0_C2) + valid_data(j,1:22)*log(p1_C2) + log(PC2);
        if PC1X > PC2X
            C=1;
        else
            C=2;
        end
            if valid_data(j,23) ~= C
                error = error+1;
            end
    end
    error_rate = error/valid_size;
    if error_rate < min
        min = error_rate;
        best_prior_C1 = PC1;
    end
    sprintf('Given sigma = %d, PC1=%f, error rate = %f \n',sigma,PC1,error_rate);
    fprintf('Given sigma = %d, PC1=%f, error rate = %f \n',sigma,PC1,error_rate);
end

best_prior_C2 = 1 - best_prior_C1;
p1 = p1_C1;
p2 = p1_C2;
pc1 = best_prior_C1;
pc2 = best_prior_C2;
