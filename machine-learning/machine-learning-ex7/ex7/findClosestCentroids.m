function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

K = size(centroids, 1);
m = size(X, 1);
idx = zeros(size(X,1), 1);

for i=1:m
    row = X(i, :);
    mindist = 0;

    for k=1:K
        dist = norm(row - centroids(k, :)) ^ 2;
        if(k == 1 || dist < mindist)
            idx(i) = k; 
            mindist = dist;
        end
    end
end

end