function [Ztrain, Zvalid, w, v] = mlptrain(train, valid, d, m, k)
% Run: 
% [Ztrain,Zvalid,w,v] = mlptrain('optdigits_train.txt', 'optdigits_valid.txt', 64, [2,4,6,8,10,12,14,16,18,20], 10);
% [Ztest] = mlptest('optdigits_test.txt',w,v,20);
% [Ztrain,Zvalid,w,v] = mlptrain('optdigits_train.txt', 'optdigits_valid.txt', 64, 2, 10);
% [Ztest] = mlptest('optdigits_test.txt',w,v,2);
% [Ztrain,Zvalid,w,v] = mlptrain('optdigits_train.txt', 'optdigits_valid.txt', 64, 3, 10);
% [Ztest] = mlptest('optdigits_test.txt',w,v,3);

train_data = importdata(train);
valid_data = importdata(valid);
test_data = importdata('optdigits_test.txt');
d=64;
X3 = test_data(:,1:d);
labels3 = test_data(:,65);
X = train_data(:,1:d);
X2 = valid_data(:,1:d);
labels = train_data(:,65);
labels2 = valid_data(:,65);
hidden = m; %m could be an array or just a number
K=k; 
ERROR_TRAIN = zeros(1,size(m,2));
ERROR_VALID = zeros(1,size(m,2));

for hid = 1:size(m,2)
H=hidden(hid); 
learning_rate = 0.001;
v = -0.01 + 0.02 * rand(K,H + 1); %10(no. of outputs) by 3(hidden units + bias)
w = -0.01 + 0.02 * rand(H, d+1);
v_up = zeros(K, H+1); 
w_up = zeros(H, d+1);
outputs = zeros(K, 1); 
y = zeros(K, 1); %depends on the k outputs 
z = ones(H + 1, 1); %h hidden units
count = 1; 
E = 0;
a=1;
ii=1;
pred = zeros(size(X,1),1);
pred2 = zeros(size(X2,1),1);

%Backpropagation 
while(ii == 1)
    E_old = E;
    E=0;
    count = 1;
    %w = zeros(H,65);
    %v = zeros(10,H+1);
    while(count <= size(X,1)) %count is the rows index
        for h = 2:(H+1)
            z(h) = 1/(1 + exp(-w(h-1,:) * [1,X(count,:)]'));
        end
        sum = 0;
        %equation (11.25)
        for i = 1:K
            outputs(i) = v(i,:) * z;
            sum = sum + exp(outputs(i));
        end
        %equation (11.26)
        for i = 1:K
            y(i) = exp(outputs(i))/sum;
        end
        %equation (11.28)
        for i = 1:K
            if (labels(count) == i-1)
                v_up(i,:) = learning_rate * (1 - y(i)) * z';
            else
                v_up(i,:) = learning_rate * (0 - y(i)) * z';
            end
        end
        %equation (11.29)
        for h = 1:H
            sum = 0;
            for j = 1:K
                if (labels(count) == j-1)
                    sum = sum + (1 - y(j)) * v(j,h+1);
                else
                    sum = sum + (0 - y(j)) * v(j,h+1); %changed from h to h+1
                end
            end
            w_up(h,:) = learning_rate * sum * z(h+1) * (1-z(h+1)) * [1,X(count,:)];
        end
        v = v + v_up; %update
        w = w + w_up; %update
        %equation (11.27)
        for i = 1:K
            if (labels(count) == i-1)
                E = E - log(y(i));
            end
        end
        count = count + 1;
    end
    %disp(abs(E_old-E));
    if a~=1 
        if (abs(E_old - E) <= 10)
            ii = 2;
        end
    end
    a=a+1;
end

Ztrain = zeros(size(X,1),H+1);
for t=1:size(X,1)
    for h = 2:(H+1)
        z(h) = 1/(1 + exp(-w(h-1,:) * [1,X(t,:)]'));
    end;
    Ztrain(t,:) = z';
    sum = 0;
    %equation (11.26)
    for i = 1:K
        outputs(i) = v(i,:) * z;
        sum = sum + exp(outputs(i));
    end
    %equation (11.26)
    for i = 1:K
        y(i) = exp(outputs(i))/sum;
    end
    [val,c] = max(y);
    pred(t) = c-1;
end
Ztrain = Ztrain(:,2:H+1);

compare = pred == labels;
error_rate = size(compare(compare==0),1)/size(compare,1);
disp('error_rate for training =');
disp(error_rate);
ERROR_TRAIN(hid) = error_rate;

Zvalid = zeros(size(X2,1),H+1);
for t=1:size(X2,1)
    for h = 2:(H+1)
        z(h) = 1/(1 + exp(-w(h-1,:) * [1,X2(t,:)]'));
    end
    Zvalid(t,:) = z';
    sum = 0;
    %equation (11.26)
    for i = 1:K
        outputs(i) = v(i,:) * z;
        sum = sum + exp(outputs(i));
    end
    %equation (11.26)
    for i = 1:K
        y(i) = exp(outputs(i))/sum;
    end
    [val,c] = max(y);
    pred2(t) = c-1;
end
Zvalid = Zvalid(:,2:H+1);

compare = pred2 == labels2;
error_rate = size(compare(compare==0),1)/size(compare,1);
disp('error_rate for validation =');
disp(error_rate);
ERROR_VALID(hid) = error_rate;

end

if (m == [2,4,6,8,10,12,14,16,18,20])
    figure
    plot(ERROR_TRAIN,hidden,'b');
    hold on;
    plot(ERROR_VALID,hidden,'r');
    xlabel('Error rates');
    ylabel('Number of hidden units');
    legend('Training Error Rates','Validation Error Rates');
end

if H==2
figure
plot(Ztrain(pred==0,1),Ztrain(pred==0,2), 'ob'); text(Ztrain(pred==0,1),Ztrain(pred==0,2),'0');hold on;
plot(Ztrain(pred==1,1),Ztrain(pred==1,2), '^r'); text(Ztrain(pred==1,1),Ztrain(pred==1,2),'1');hold on;
plot(Ztrain(pred==2,1),Ztrain(pred==2,2), 'xc'); text(Ztrain(pred==2,1),Ztrain(pred==2,2),'2');hold on;
plot(Ztrain(pred==3,1),Ztrain(pred==3,2), 'hg'); text(Ztrain(pred==3,1),Ztrain(pred==3,2),'3');hold on;
plot(Ztrain(pred==4,1),Ztrain(pred==4,2), 'sk'); text(Ztrain(pred==4,1),Ztrain(pred==4,2),'4');hold on;
plot(Ztrain(pred==5,1),Ztrain(pred==5,2), 'dy'); text(Ztrain(pred==5,1),Ztrain(pred==5,2),'5');hold on;
plot(Ztrain(pred==6,1),Ztrain(pred==6,2), '*g'); text(Ztrain(pred==6,1),Ztrain(pred==6,2),'6');hold on;
plot(Ztrain(pred==7,1),Ztrain(pred==7,2), '*k'); text(Ztrain(pred==7,1),Ztrain(pred==7,2),'7');hold on;
plot(Ztrain(pred==8,1),Ztrain(pred==8,2), '<m'); text(Ztrain(pred==8,1),Ztrain(pred==8,2),'8');hold on;
plot(Ztrain(pred==9,1),Ztrain(pred==9,2), '+r'); text(Ztrain(pred==9,1),Ztrain(pred==9,2),'9');

figure
plot(Zvalid(pred2==0,1),Zvalid(pred2==0,2), 'ob'); text(Zvalid(pred2==0,1),Zvalid(pred2==0,2),'0');hold on;
plot(Zvalid(pred2==1,1),Zvalid(pred2==1,2), '^r'); text(Zvalid(pred2==1,1),Zvalid(pred2==1,2),'1');hold on;
plot(Zvalid(pred2==2,1),Zvalid(pred2==2,2), 'xc'); text(Zvalid(pred2==2,1),Zvalid(pred2==2,2),'2');hold on;
plot(Zvalid(pred2==3,1),Zvalid(pred2==3,2), 'hg'); text(Zvalid(pred2==3,1),Zvalid(pred2==3,2),'3');hold on;
plot(Zvalid(pred2==4,1),Zvalid(pred2==4,2), 'sk'); text(Zvalid(pred2==4,1),Zvalid(pred2==4,2),'4');hold on;
plot(Zvalid(pred2==5,1),Zvalid(pred2==5,2), 'dy'); text(Zvalid(pred2==5,1),Zvalid(pred2==5,2),'5');hold on;
plot(Zvalid(pred2==6,1),Zvalid(pred2==6,2), '*g'); text(Zvalid(pred2==6,1),Zvalid(pred2==6,2),'6');hold on;
plot(Zvalid(pred2==7,1),Zvalid(pred2==7,2), '*k'); text(Zvalid(pred2==7,1),Zvalid(pred2==7,2),'7');hold on;
plot(Zvalid(pred2==8,1),Zvalid(pred2==8,2), '<m'); text(Zvalid(pred2==8,1),Zvalid(pred2==8,2),'8');hold on;
plot(Zvalid(pred2==9,1),Zvalid(pred2==9,2), '+r'); text(Zvalid(pred2==9,1),Zvalid(pred2==9,2),'9');
end

if H==3
figure
plot3(Ztrain(pred==0,1),Ztrain(pred==0,2),Ztrain(pred==0,3), 'ob'); text(Ztrain(pred==0,1),Ztrain(pred==0,2),Ztrain(pred==0,3),'0');hold on;
plot3(Ztrain(pred==1,1),Ztrain(pred==1,2),Ztrain(pred==1,3), '^r'); text(Ztrain(pred==1,1),Ztrain(pred==1,2),Ztrain(pred==1,3),'1');hold on;
plot3(Ztrain(pred==2,1),Ztrain(pred==2,2),Ztrain(pred==2,3), 'xc'); text(Ztrain(pred==2,1),Ztrain(pred==2,2),Ztrain(pred==2,3),'2');hold on;
plot3(Ztrain(pred==3,1),Ztrain(pred==3,2),Ztrain(pred==3,3), 'hg'); text(Ztrain(pred==3,1),Ztrain(pred==3,2),Ztrain(pred==3,3),'3');hold on;
plot3(Ztrain(pred==4,1),Ztrain(pred==4,2),Ztrain(pred==4,3), 'sk'); text(Ztrain(pred==4,1),Ztrain(pred==4,2),Ztrain(pred==4,3),'4');hold on;
plot3(Ztrain(pred==5,1),Ztrain(pred==5,2),Ztrain(pred==5,3), 'dy'); text(Ztrain(pred==5,1),Ztrain(pred==5,2),Ztrain(pred==5,3),'5');hold on;
plot3(Ztrain(pred==6,1),Ztrain(pred==6,2),Ztrain(pred==6,3), '*g'); text(Ztrain(pred==6,1),Ztrain(pred==6,2),Ztrain(pred==6,3),'6');hold on;
plot3(Ztrain(pred==7,1),Ztrain(pred==7,2),Ztrain(pred==7,3), '*k'); text(Ztrain(pred==7,1),Ztrain(pred==7,2),Ztrain(pred==7,3),'7');hold on;
plot3(Ztrain(pred==8,1),Ztrain(pred==8,2),Ztrain(pred==8,3), '<m'); text(Ztrain(pred==8,1),Ztrain(pred==8,2),Ztrain(pred==8,3),'8');hold on;
plot3(Ztrain(pred==9,1),Ztrain(pred==9,2),Ztrain(pred==9,3), '+r'); text(Ztrain(pred==9,1),Ztrain(pred==9,2),Ztrain(pred==9,3),'9');

figure
plot3(Zvalid(pred2==0,1),Zvalid(pred2==0,2),Zvalid(pred2==0,3), 'ob'); text(Zvalid(pred2==0,1),Zvalid(pred2==0,2),Zvalid(pred2==0,3),'0');hold on;
plot3(Zvalid(pred2==1,1),Zvalid(pred2==1,2),Zvalid(pred2==1,3), '^r'); text(Zvalid(pred2==1,1),Zvalid(pred2==1,2),Zvalid(pred2==1,3),'1');hold on;
plot3(Zvalid(pred2==2,1),Zvalid(pred2==2,2),Zvalid(pred2==2,3), 'xc'); text(Zvalid(pred2==2,1),Zvalid(pred2==2,2),Zvalid(pred2==2,3),'2');hold on;
plot3(Zvalid(pred2==3,1),Zvalid(pred2==3,2),Zvalid(pred2==3,3), 'hg'); text(Zvalid(pred2==3,1),Zvalid(pred2==3,2),Zvalid(pred2==3,3),'3');hold on;
plot3(Zvalid(pred2==4,1),Zvalid(pred2==4,2),Zvalid(pred2==4,3), 'sk'); text(Zvalid(pred2==4,1),Zvalid(pred2==4,2),Zvalid(pred2==4,3),'4');hold on;
plot3(Zvalid(pred2==5,1),Zvalid(pred2==5,2),Zvalid(pred2==5,3), 'dy'); text(Zvalid(pred2==5,1),Zvalid(pred2==5,2),Zvalid(pred2==5,3),'5');hold on;
plot3(Zvalid(pred2==6,1),Zvalid(pred2==6,2),Zvalid(pred2==6,3), '*g'); text(Zvalid(pred2==6,1),Zvalid(pred2==6,2),Zvalid(pred2==6,3),'6');hold on;
plot3(Zvalid(pred2==7,1),Zvalid(pred2==7,2),Zvalid(pred2==7,3), '*k'); text(Zvalid(pred2==7,1),Zvalid(pred2==7,2),Zvalid(pred2==7,3),'7');hold on;
plot3(Zvalid(pred2==8,1),Zvalid(pred2==8,2),Zvalid(pred2==8,3), '<m'); text(Zvalid(pred2==8,1),Zvalid(pred2==8,2),Zvalid(pred2==8,3),'8');hold on;
plot3(Zvalid(pred2==9,1),Zvalid(pred2==9,2),Zvalid(pred2==9,3), '+r'); text(Zvalid(pred2==9,1),Zvalid(pred2==9,2),Zvalid(pred2==9,3),'9');
end
    


