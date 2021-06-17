function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

C_list= [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

predictions = zeros(length(yval));
predict_err= zeros(length(yval));

matrix_rslt = zeros(length(C_list)*length(sigma_list), 3);

row=1;

for C_val=C_list,
  for sigma_val=sigma_list,
    model= svmTrain(X, y, C_val, @(x1, x2) gaussianKernel(x1, x2, sigma_val)); 
    predictions = svmPredict(model, Xval);
    predict_err = mean(double(predictions ~= yval));
    matrix_rslt(row, :) = [C_val, sigma_val, predict_err];
    row = row + 1;
  endfor
endfor
[err_value idx] = min(matrix_rslt(:,3));
%disp(err_value);
%disp(idx);

C = matrix_rslt(idx, 1);
sigma = matrix_rslt(idx, 2);

%disp('C, sigma, errors matrix');
%disp(matrix_rslt);

disp(C) , disp(sigma); 

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
