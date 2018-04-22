function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

m = length(y);              % number of training examples
h = X * theta;              % hypothesis
l2 = sum((h - y) .^ 2);     % l2 norm
J = l2 / (2 * m);           % cost

end
