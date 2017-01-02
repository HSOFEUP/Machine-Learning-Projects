function alp = kernPercGD(training_data);

%train_data = importdata('optdigits79_train.txt');
train_data = importdata(training_data);
d = size(train_data,2);
X = train_data(:,1:d-1);
theclass = train_data(:,d);
N = size(X,1);
alp = zeros(N,1);

for iter=1:5
    rule = 0;
    for i = 1:N
        rule = 0;
        for j = 1:N
            kernel = exp(-norm(X(j,:)-X(i,:))^2/4); %S^2 = 4
            rule = rule + alp(j) * theclass(j) * kernel;
        end
        rule = rule * theclass(i);
        if rule <= 0
            alp(i) = alp(i) + 1; 
        end
        %fprintf('Current index = %d\n',i);
    end
    fprintf('Current Iteration = %d\n',iter);
end


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
fprintf('For the %s dataset, the error rate = %f \n',training_data,error_rate);