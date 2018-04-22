function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% Notes: X - num_movies x num_features matrix of movie features
%        Theta - num_users x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
            

cost = sum(sum(((X * Theta' - Y) .^ 2) .* R)) / 2;
reg_theta = sum(sum(Theta .^ 2)) * lambda / 2;
reg_X = sum(sum(X .^ 2)) * lambda / 2;  
J = cost + reg_theta + reg_X;

X_grad = (((X * Theta' - Y) .* R) * Theta) + (lambda * X);
Theta_grad = (((X * Theta' - Y) .* R)' * X) + (lambda * Theta);

grad = [X_grad(:); Theta_grad(:)];

end
