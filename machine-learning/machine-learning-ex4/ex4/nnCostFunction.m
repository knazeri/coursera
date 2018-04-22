function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

% (hidden_layer_size x input_layer_size+1)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

% (num_labels x hidden_layer_size+1)
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


% Part 1: Feedforward and Cost Function


m = size(X, 1);                         % number of training examples
n = size(X, 2);                         % number of features

a1 = [ones(m, 1) X];                    % (m x n+1) matrix: add the intercept column

z2 = a1 * Theta1';                      % (m x hidden_layer_size) matrix
a2 = [ones(m, 1) sigmoid(z2)];          % (m x hidden_layer_size+1) matrix: sigmoid + intercept column

z3 = a2 * Theta2';                      % (m x num_labels) matrix
a3 = sigmoid(z3);                       % apply sigmoid

h = a3;                                 % hypothesis
Y = zeros(m, num_labels);               % binary vector of 1's and 0's

for i = 1:m
    Y(i, y(i)) = 1;
end

cost = sum(sum(-Y .* log(h) - (1 - Y) .* log(1 - h))) / m;
reg1 = sum(sum(Theta1(:,2:end) .^ 2));
reg2 = sum(sum(Theta2(:,2:end) .^ 2));
cost_reg = (reg1 + reg2) * lambda / (2 * m);
         
J = cost + cost_reg;



% Part 2: Implement the backpropagation

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for t = 1:m

    %% =============== High Performance Implementation ===============

    a1_t = a1(t,:)';                    % (n+1 x 1) t-th training example
    z2_t = z2(t,:)';                    % (hidden_layer_size+1 x 1)
    a2_t = a2(t,:)';                    % (hidden_layer_size+1 x 1)
    a3_t = a3(t,:)';                    % (num_labels x 1)

    d3_t = a3_t - Y(t,:)';                                      % (num_labels x 1) output error
    d2_t = (Theta2' * d3_t) .* [1; sigmoidGradient(z2_t)];      % (hidden_layer_size+1 x 1)
    d2_t = d2_t(2:end);                                         % taking of the bias
 

    %% =============== Poor Performance Implementation ===============

    % a1_t = [1; X(t,:)'];
    % z2_t = Theta1 * a1_t;
    % a2_t = [1; sigmoid(z2_t)];
    % z3_t = Theta2 * a2_t;
    % a3_t = sigmoid(z3_t);

    % d3_t = a3_t - Y(t,:)';
    % d2_t = (Theta2' * d3_t) .* [1; sigmoidGradient(z2_t)];
    % 

    % =========================================================================

    Theta1_grad = Theta1_grad + d2_t * a1_t';
    Theta2_grad = Theta2_grad + d3_t * a2_t';
end


Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;



% Part 2: Implement the regularization

Theta1_grad_reg = [zeros(size(Theta1, 1), 1) Theta1(:,2:end)] * lambda / m;
Theta2_grad_reg = [zeros(size(Theta2, 1), 1) Theta2(:,2:end)] * lambda / m;

Theta1_grad = Theta1_grad + Theta1_grad_reg;
Theta2_grad = Theta2_grad + Theta2_grad_reg;



% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end