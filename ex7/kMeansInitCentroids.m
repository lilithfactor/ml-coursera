% Function to return K initial centroids to be
% used with the K-Means on the dataset X
function centroids = kMeansInitCentroids(X, K) % initializes K centroids that are to be 
    % used in K-Means on the dataset X
    centroids = zeros(K, size(X, 2));
    % initializing samples to be random examples
    % randomly reorder indices of examples
    randidx = randperm(size(X, 1));
    % take the first K examples
    centroids = X(randidx(1:K), :);
end