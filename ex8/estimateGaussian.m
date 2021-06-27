% input X is the dataset with each n-dimensional data point in one row
function [mu sigma2] = estimateGaussian(X) % estimates the parameters of a 
    % Gaussian distribution using the data in X
    [m, n] = size(X);
    mu = zeros(n, 1);
    sigma2 = zeros(n, 1);
    % mean
    mu = ((1/m)*sum(X))';
    % sigma^2
    sigma2 = ((1/m)*sum((X-mu').^2))';
end