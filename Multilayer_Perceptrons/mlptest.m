function [Ztest] = mlptest(test,w,v,m)

test_data = importdata(test);
d = size(w,2)-1;
X = test_data(:,1:d);
labels = test_data(:,65);
K = size(v,1);
outputs = zeros(K, 1); 
y = zeros(K, 1);
z = ones(m + 1, 1);
pred = zeros(size(X,1),1);

Ztest = zeros(size(X,1),m+1);
for t=1:size(X,1)
    for h = 2:(m+1)
        z(h) = 1/(1 + exp(-w(h-1,:) * [1,X(t,:)]'));
    end;
    Ztest(t,:) = z';
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
Ztest = Ztest(:,2:m+1);

compare = pred == labels;
error_rate = size(compare(compare==0),1)/size(compare,1);
disp('error_rate for testing =');
disp(error_rate);

if m==2
figure
plot(Ztest(pred==0,1),Ztest(pred==0,2), 'ob'); text(Ztest(pred==0,1),Ztest(pred==0,2),'0');hold on;
plot(Ztest(pred==1,1),Ztest(pred==1,2), '^r'); text(Ztest(pred==1,1),Ztest(pred==1,2),'1');hold on;
plot(Ztest(pred==2,1),Ztest(pred==2,2), 'xc'); text(Ztest(pred==2,1),Ztest(pred==2,2),'2');hold on;
plot(Ztest(pred==3,1),Ztest(pred==3,2), 'hg'); text(Ztest(pred==3,1),Ztest(pred==3,2),'3');hold on;
plot(Ztest(pred==4,1),Ztest(pred==4,2), 'sk'); text(Ztest(pred==4,1),Ztest(pred==4,2),'4');hold on;
plot(Ztest(pred==5,1),Ztest(pred==5,2), 'dy'); text(Ztest(pred==5,1),Ztest(pred==5,2),'5');hold on;
plot(Ztest(pred==6,1),Ztest(pred==6,2), '*g'); text(Ztest(pred==6,1),Ztest(pred==6,2),'6');hold on;
plot(Ztest(pred==7,1),Ztest(pred==7,2), '*k'); text(Ztest(pred==7,1),Ztest(pred==7,2),'7');hold on;
plot(Ztest(pred==8,1),Ztest(pred==8,2), '<m'); text(Ztest(pred==8,1),Ztest(pred==8,2),'8');hold on;
plot(Ztest(pred==9,1),Ztest(pred==9,2), '+r'); text(Ztest(pred==9,1),Ztest(pred==9,2),'9');
end

if m==3
figure
plot3(Ztest(pred==0,1),Ztest(pred==0,2),Ztest(pred==0,3), 'ob'); text(Ztest(pred==0,1),Ztest(pred==0,2),Ztest(pred==0,3),'0');hold on;
plot3(Ztest(pred==1,1),Ztest(pred==1,2),Ztest(pred==1,3), '^r'); text(Ztest(pred==1,1),Ztest(pred==1,2),Ztest(pred==1,3),'1');hold on;
plot3(Ztest(pred==2,1),Ztest(pred==2,2),Ztest(pred==2,3), 'xc'); text(Ztest(pred==2,1),Ztest(pred==2,2),Ztest(pred==2,3),'2');hold on;
plot3(Ztest(pred==3,1),Ztest(pred==3,2),Ztest(pred==3,3), 'hg'); text(Ztest(pred==3,1),Ztest(pred==3,2),Ztest(pred==3,3),'3');hold on;
plot3(Ztest(pred==4,1),Ztest(pred==4,2),Ztest(pred==4,3), 'sk'); text(Ztest(pred==4,1),Ztest(pred==4,2),Ztest(pred==4,3),'4');hold on;
plot3(Ztest(pred==5,1),Ztest(pred==5,2),Ztest(pred==5,3), 'dy'); text(Ztest(pred==5,1),Ztest(pred==5,2),Ztest(pred==5,3),'5');hold on;
plot3(Ztest(pred==6,1),Ztest(pred==6,2),Ztest(pred==6,3), '*g'); text(Ztest(pred==6,1),Ztest(pred==6,2),Ztest(pred==6,3),'6');hold on;
plot3(Ztest(pred==7,1),Ztest(pred==7,2),Ztest(pred==7,3), '*k'); text(Ztest(pred==7,1),Ztest(pred==7,2),Ztest(pred==7,3),'7');hold on;
plot3(Ztest(pred==8,1),Ztest(pred==8,2),Ztest(pred==8,3), '<m'); text(Ztest(pred==8,1),Ztest(pred==8,2),Ztest(pred==8,3),'8');hold on;
plot3(Ztest(pred==9,1),Ztest(pred==9,2),Ztest(pred==9,3), '+r'); text(Ztest(pred==9,1),Ztest(pred==9,2),Ztest(pred==9,3),'9');
end
