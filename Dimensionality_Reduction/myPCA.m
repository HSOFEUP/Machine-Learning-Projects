function myPCA(faces_data)
%Run: myPCA('faces.txt');
faces = importdata(faces_data);

%implementing PCA
me = mean(faces);
centered = bsxfun(@minus, faces, me); %me remains the same
covariancematrix=cov(faces);
[V,D] = eig(covariancematrix);
D=diag(D);
[D, order] = sort(D, 'descend'); 
sorted_eigenval=V(:,order); %sorted eigenvalues with corresponding eigenvectors
finaldata =centered*sorted_eigenval; 

%visualizing part 1  
figure
subplot(1, 5, 1);
imagesc(reshape(sorted_eigenval(:,1),60,64));
subplot(1, 5, 2);
imagesc(reshape(sorted_eigenval(:,2),60,64));

%visualizing part 2
d = [10 50 100];
p=3; %subplot position; p=1,2 are used for the previous part
for i=1:length(d)
    first_image = faces(1,:);
    centered_image = bsxfun(@minus, first_image, mean(first_image));
    reconstruct = centered_image*sorted_eigenval(:,1:d(i));
    backproject = reconstruct*(sorted_eigenval(:,1:d(i)))';
    backproject = backproject+mean(first_image);
    subplot(1,5,p);
    imagesc(reshape(backproject,60,64));
    p=p+1;
end