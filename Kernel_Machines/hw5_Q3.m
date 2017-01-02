function hw5_Q3()

rng(1); % For reproducibility
r = sqrt(rand(100,1)); % Radius
t = 2*pi*rand(100,1); % Angle
data1 = [r.*cos(t), r.*sin(t)]; % Points
r2 = sqrt(3*rand(100,1)+2); % Radius
t2 = 2*pi*rand(100,1); % Angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points

%Visualize
figure; 
plot(data1(:,1),data1(:,2),'r.','MarkerSize',15);
hold on 
plot(data2(:,1),data2(:,2),'b.','MarkerSize',15);
ezpolar(@(x)1);ezpolar(@(x)2);
axis equal
hold off

%Combine
data3 = [data1;data2];
theclass = ones(200,1);
theclass(1:100) = -1;
dlmwrite('data3.txt',[data3 theclass]);
%training for data3
alp = kernPercGD('data3.txt');

figure;
h(1:2) = gscatter(data3(:,1),data3(:,2),theclass,'rg','+*');
hold on
ezpolar(@(x)1);
hold on

%Play with the boxconstraint parameter
SVMStruct = svmtrain(data3,theclass,'kernel_function','rbf','BoxConstraint',1,'showplot',true);
hold on
SVMStruct = svmtrain(data3,theclass,'kernel_function','rbf','BoxConstraint',5,'showplot',true);
hold on
SVMStruct = svmtrain(data3,theclass,'kernel_function','rbf','BoxConstraint',10,'showplot',true);
hold off

%training and testing for optdigits49
alp = kernPercGD('optdigits49_train.txt');
test_data = importdata('optdigits49_test.txt');
d = size(test_data,2);
X = test_data(:,1:d-1);
theclass = test_data(:,d);
N = size(X,1);
labels = zeros(N,1);
for i = 1:N
    rule = 0;
    for j = 1:N
        kernel = exp(-norm(X(j,:)-X(i,:))^2/4);
        rule = rule + alp(j) * theclass(j) * kernel;
    end
    if rule < 0
        labels(i) = -1;
    else
        labels(i) = 1;
    end
end
compare = labels == theclass;
error_rate = 1 - (sum(compare)/N);
fprintf('For the %s dataset, the error rate = %f \n','optdigits49_test.txt',error_rate);

%training and testing for optdigits79
alp = kernPercGD('optdigits79_train.txt');
test_data = importdata('optdigits79_test.txt');
d = size(test_data,2);
X = test_data(:,1:d-1);
theclass = test_data(:,d);
N = size(X,1);
labels = zeros(N,1);
for i = 1:N
    rule = 0;
    for j = 1:N
        kernel = exp(-norm(X(j,:)-X(i,:))^2/4);
        rule = rule + alp(j) * theclass(j) * kernel;
    end
    if rule < 0
        labels(i) = -1;
    else
        labels(i) = 1;
    end
end
compare = labels == theclass;
error_rate = 1 - (sum(compare)/N);
fprintf('For the %s dataset, the error rate = %f \n','optdigits79_test.txt',error_rate);
