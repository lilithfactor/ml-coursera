% Function to take a data matrix X (size m x 1) and
% maps each example into its polynomial features where
% X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p]
function [X_poly] = polyFeatures(X, p) % Maps X (1D vector) into the p-th power
    % variables to be returned
    X_poly = zeros(numel(X), p);
    % vectorized implementation
    X_poly(:, 1:p) = X(:, 1).^(1:p);
end