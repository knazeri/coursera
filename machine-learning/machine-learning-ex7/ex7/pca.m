function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%


[m, n] = size(X);

sigma = (X' * X) / m;           % covariance matrix (n x n)
[U, S, V] = svd(sigma);         % eigen decomposition using SVD

end