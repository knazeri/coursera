function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y);                              % number of training examples
h = X * theta;                              % hypothesis
theta(1) = 0;                               % not regularizing theta_0
l = sum((h - y) .^ 2);                      % linear regression cost
r = sum(theta .^ 2);                        % regularization (not regularizing theta_0)
J = l / (2 * m) + lambda * r / (2 * m);     % cost function


% Regularized linear regression gradient
grad = (X' * (h - y)) / m + lambda * theta / m; 
grad = grad(:);

end