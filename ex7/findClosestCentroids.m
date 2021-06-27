% Function to return the closest centroids
% in idx for a dataset X where each row is a single example
function idx = findClosestCentroids(X, centroids) % computes the centroid memberships for every example
    % Set K
    K = size(centroids, 1);
    % variables to be returned
    idx = zeros(size(X,1), 1);
    m = size(X, 1);
    % DIMENSIONS:
    % centroids = K x no. of features = 3 x 2
    for i = 1:m
        temp = zeros(K,1);
        for j = 1:K
          temp(j) = sqrt( sum( (X(i,:) - centroids(j,:) ).^2) );
        end
        % discarding the smallest values and keeping the indexes
        [~, idx(i)] = min(temp);
    end
end