function Bayes_Testing(test_data, p1, p2, pc1, pc2);
%Run: Bayes_Testing('SPECT_test.txt', p1, p2, pc1, pc2);

test_data = importdata(test_data);
test_size = size(test_data, 1);

PC1=pc1;
PC2=1-PC1;
p1_C1 = p1;
p0_C1 = 1 - p1;
p1_C2 = p2;
p0_C2 = 1 - p2;
error = 0;
for j=1:test_size
    PC1X = (1-test_data(j,1:22))*log(p0_C1) + test_data(j,1:22)*log(p1_C1) + log(PC1);
    PC2X = (1-test_data(j,1:22))*log(p0_C2) + test_data(j,1:22)*log(p1_C2) + log(PC2);
    if PC1X > PC2X
        C=1;
    else
        C=2;
    end
    if test_data(j,23) ~= C
        error = error+1;
    end
end
error_rate = error/test_size;
sprintf('error rate = %f\n',error_rate);
fprintf('error rate = %f\n',error_rate);
