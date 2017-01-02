function [h,m,Q] = EMG(img,k)
% Run: [h,m,Q] = EMG('stadium.bmp,k) for k=4,8,12;
tic
[X,map] = imread(img);
img_rgb = ind2rgb(X,map);
img_double = im2double(img_rgb);
A = reshape(img_double,[],3);
n = size(A,1);
d = size(A,2);
[idx,m] = kmeans(A,k,'EmptyAction','singleton'); %idx is initial labels; C is initial cluster centers

%initializing PI
PI = zeros(1,k);
for i=1:k
    PI(i) = size(A(idx==i),1)/n;
end

%initializing SIGMA
SIGMA = zeros(d,d,k);
for i=1:k
    SIGMA(:,:,i) = cov(A(idx==i,:));
end

%creating h matrix
h = zeros(n,k);

%initializing Q vector and its index
Q = zeros(200,1);
ii = 1;

%initializing iteration 
iter = 1;

%running 100 iterations
while (iter<=100)
    
    disp(iter); %displaying the number of iterations
    for i=1:k
        centered = bsxfun(@minus,A,m(i,:));
        temp = centered/SIGMA(:,:,i).*centered;
        temp2 = sum(temp,2);
        temp2 = exp(-0.5*temp2);
        temp2 = PI(i)*((det(SIGMA(:,:,i)))^(-1/2))*temp2;
        h(:,i) = temp2;
    end
    count = sum(h,2);
    h = bsxfun(@rdivide,h,count);
    
    %Calculating Q for E-step
    q = 0;
    for i=1:k
        Di = mvnpdf(A,m(i,:),SIGMA(:,:,i));
        Di(Di==0) = 10^(-20);
        temp = log(PI(i))+log(Di);
        temp = h(:,i)' * temp;
        q = q + temp;
    end
    Q(ii) = q;
    ii = ii + 1;
    
    %M-step
    %updating PI
    %testing PI2 results: PI = P2
    PI = sum(h)/n; 
    
    %updating m
    old_m = m; %saving old m
    m = h'*A;
    count = sum(h);
    trans = m';
    m = bsxfun(@ldivide,count,trans);
    m = m';
    
    %updating SIGMA
    for i=1:k
        temp = zeros(3);
        for t = 1:n
            temp = temp + h(t,i) * ((A(t,:)-m(i,:))' * (A(t,:)-m(i,:)));
        end
        count = sum(h(:,i));
        if (count ~= 0)
            temp = temp/count;
        end
        SIGMA(:,:,i) = temp;
    end
    
    %Calculating Q for M-step
    q = 0;
    for i=1:k
        Di = mvnpdf(A,m(i,:),SIGMA(:,:,i));
        Di(Di==0) = 10^(-20);
        temp = log(PI(i))+log(Di);
        temp = h(:,i)' * temp;
        q = q + temp;
    end
    Q(ii) = q;
    ii = ii + 1;
    
    iter = iter + 1; %iter increment
    
end

%predicting cluster (G) of each sample belongs to, based on the final h matrix
G = zeros(n,1);
for i=1:n
    [Maxi,I] = max(h(i,:));
    G(i) = I;
end

%replacing the existing pixels to the mean corresponding to the respective cluster they belong to
compressed = zeros(n,d);
for t=1:n
    compressed(t,:) = m(G(t),:);
end

%reshaping it back to 3D matrix then visualize
compressed_3D = reshape(compressed,67,200,3);
image(compressed_3D);
figure;

toc

plot(Q(1:2:length(Q)),1:1:100,'b');
hold on;
plot(Q(2:2:length(Q)),1:1:100,'r');
title(['Expected complete log-likelihood function value vs Iteration number for k=',num2str(k),'']);
xlabel('Expected complete log-likelihood function value');
ylabel('Iteration number');
legend('E-step','M-step');
