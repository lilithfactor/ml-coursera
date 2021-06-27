% Function to return the new centroids by 
% computing the means of the data points assigned to each centroid
function centroids = computeCentroids(X, idx, K) % returns the new centroids by 
    % computing the means of the data points assigned to each centroid
    [m n] = size(X);

    % variables to be returned
    centroids = zeros(K, n);

    % X dim:  m x n
    % centroids dim: K x n

    % for all cluster centroids
    for i = 1:K
        %indexes of all the input which belongs to cluster j
        idx_i = find(idx==i);       
        % calculating mean using built-in function
        centroids(i,:) = mean(X(idx_i,:)); 
    end
end