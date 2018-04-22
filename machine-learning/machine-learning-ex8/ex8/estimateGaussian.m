function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

[m, n] = size(X);

mu = (sum(X) / m)';                     % estimate the mean
dif = bsxfun(@minus, X', mu);
sigma2 = sum(dif .^ 2, 2) ./ m;         % estimate the variance

end
