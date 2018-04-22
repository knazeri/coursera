function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y);                                              % number of training examples
ht = sigmoid(X * theta);                                    % hypothesis

cost = sum(-y .* log(ht) - (1 - y) .* log(1 - ht)) / m;     % logistic regression cost
cost_reg = sum(theta(2:end) .^ 2) * lambda / (2 * m);       % cost regularization

grad = (X' * (ht - y)) / m;                                 % gradient
grad_reg = [0; theta(2:end)] * lambda / m;                  % gradient regularization

J = cost + cost_reg;                                        % total cost
grad = grad + grad_reg;                                     % total regularization

end
