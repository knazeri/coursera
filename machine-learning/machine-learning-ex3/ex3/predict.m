function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)


m = size(X, 1);                     % number of training examples
n = size(X, 2);                     % number of features
s2 = size(Theta1, 1);               % number of layer 2 units - Theta1: (s2 x n+1)
k = size(Theta2, 1);                % number of labels - Theta2: (k x s2+1)


a1 = [ones(m, 1) X];                % (m x n+1) matrix: add the intercept column

z2 = a1 * Theta1';                  % (m x s2) matrix
a2 = [ones(m, 1) sigmoid(z2)];      % (m x s2+1) matrix: apply sigmoid add the intercept column

z3 = a2 * Theta2';                  % (m x k) matrix
a3 = sigmoid(z3);                   % apply sigmoid

[M, p] = max(a3, [], 2);            % get index of the max element of each row

end
