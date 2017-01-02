function [m1,m2,S1,S2,SS,SSS,alpha1,alpha2] = Probelm1(train_data,test_data);
%Run: [m1,m2,S1,S2,SS,SSS,alpha1,alpha2] = Problem1('training_data.txt','test_data.txt');
train_data = importdata('training_data.txt');
test_data = importdata('test_data.txt');  
train_cols = size(train_data,2);
test_cols = size(test_data,2);
class1 = train_data(train_data(:,train_cols)==1,1:train_cols-1);
class2 = train_data(train_data(:,train_cols)==2,1:train_cols-1);
class1_size = size(class1, 1);
class2_size = size(class2, 1);
PC1 = 0.4;
PC2 = 0.6;

%implementation for part a
m1 = mean(class1); 
m2 = mean(class2);
S1 = cov(class1);
S2= cov(class2);
disp('u1 =');
disp(m1);
disp('u2 =');
disp(m2);
disp('S1 for part (a) =');
disp(S1);
disp('S2 for part (a) =');
disp(S2);

error = 0;
for i=1:size(test_data,1)
    centered1 = bsxfun(@minus, test_data(i,1:8), m1); 
    centered2 = bsxfun(@minus, test_data(i,1:8), m2); 
    S = PC1*S1 + PC2*S2; %S is based on S1 and S2 are independent (5.21)
    g1 =  - centered1*1/2*inv(S)*centered1' + log(PC1); %(5.19)
    g2 =  - centered2*1/2*inv(S)*centered2' + log(PC2);
    if g1>g2
        C = 1;
    else
        C=2;
    end
    if C ~= test_data(i,test_cols)
        error = error+1;
    end
end
error_rate = error/size(test_data,1);
fprintf('When S1 and S2 are independent, error rate = %f \n\n',error_rate);

%implementation for part b
SS = cov(train_data(:,1:train_cols-1));
SSS = SS; %S1=S2
disp('S1 for part (b)');
disp(SS);
disp('S2 for part (b)');
disp(SSS);
error2 = 0;
for i=1:size(test_data,1)
    centered1 = bsxfun(@minus, test_data(i,1:8), m1); 
    centered2 = bsxfun(@minus, test_data(i,1:8), m2); 
    S = PC1*SS + PC2*SS; %S is based on S1 = S2
    g1 =  - centered1*1/2*inv(S)*centered1' + log(PC1); %(5.19)
    g2 =  - centered2*1/2*inv(S)*centered2' + log(PC2);
    if g1>g2
        C = 1;
    else
        C=2;
    end
    if C ~= test_data(i,test_cols)
        error2 = error2+1;
    end
end
error_rate2 = error2/size(test_data,1);
fprintf('When S1 and S2 are equal, error rate = %f \n\n',error_rate2);

%Implementation for part c
alpha1 = mean(var(class1));
alpha2 = mean(var(class2));
alpha1 = alpha1*eye(8,8);
alpha2 = alpha2*eye(8,8);
disp('alpha1 =');
disp(alpha1);
disp('alpha2 =');
disp(alpha2);
error3 = 0;
for i=1:size(test_data,1)
    centered1 = bsxfun(@minus, test_data(i,1:8), m1); 
    centered2 = bsxfun(@minus, test_data(i,1:8), m2); 
    S = PC1*alpha1 + PC2*alpha2; %S is based on S1 = S2
    g1 =  - centered1*1/2*inv(S)*centered1' + log(PC1); %(5.19)
    g2 =  - centered2*1/2*inv(S)*centered2' + log(PC2);
    if g1>g2
        C = 1;
    else
        C=2;
    end
    if C ~= test_data(i,test_cols)
        error3 = error3+1;
    end
end
error_rate3 = error3/size(test_data,1);
fprintf('When S1 and S2 are diagonal, error rate = %f \n\n',error_rate3);