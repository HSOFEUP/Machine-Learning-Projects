function class = myKNN(traindata, testdata, k) %k is an array of [1 3 5 7]
%Run: c = myKNN('optdigits_train.txt','optdigits_test.txt',[1 3 5 7]);
     class = [];
     for index =1:length(k) %iterate through k={1,3,5,7}
        n=k(index);
        train_data = importdata(traindata);
        test_data = importdata(testdata);
        train_digit = train_data(:,size(train_data,2));
        train_data = train_data(:,1:size(train_data,2)-1);
        test_digit = test_data(:,size(test_data,2));
        test_data = test_data(:,1:size(test_data,2)-1);
        train_size = size(train_data,1);
        test_size = size(test_data,1);
        
        % Calculating the distance 
        distance = pdist2(test_data, train_data, 'euclidean');
        [distance, idx] = sort(distance,2,'ascend'); % Indices of nearest neighbors
        distance = distance(:,1:n);
        idx = idx(:,1:n);
        labels = idx;
        
        for i=1:test_size
            for j=1:n
                labels(i,j) = train_digit(idx(i,j));
            end
        end
        
        %evaluating error
        compare = mode(labels,2) == test_digit;
        error = size(compare(compare==0),1)/size(compare,1);
        class = cat(2,class,mode(labels,2));
        fprintf('The error rate for k = %d is %f\n',n,error);
     end