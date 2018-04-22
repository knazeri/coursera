function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y);                                          % number of training examples
n = length(theta);                                      % number of features
f = theta(2:n);                                         % features used for regularization
h = sigmoid(X * theta);                                 % hypothesis
l = sum(-y .* log(h) - (1 - y) .* log(1 - h)) / m;      % logistic regression
r = sum(f .^ 2) * lambda / (2 * m);                     % regularization
J = l + r;                                              % cost
grad = (X' * (h - y) + [0; f] * lambda) / m;            % gradient

end
