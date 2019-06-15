function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
Ck = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
sigmak = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
errors = zeros(size(Ck,1),size(sigmak,1));

for i = 1: size(Ck,1),
for j = 1: size(sigmak,1),
model= svmTrain(X, y, Ck(i), @(x1, x2) gaussianKernel(x1, x2, sigmak(j)));
predictions = svmPredict(model, Xval);
errors(i,j) = mean(double(predictions ~= yval));
end
end
[h k] = min(errors(:));
j = ceil(k/size(Ck,1));
if mod(k,size(Ck,1))==0,
i=size(Ck);
else,
i = mod(k,size(Ck,1));
end
C = Ck(i)
sigma = sigmak(j)


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
