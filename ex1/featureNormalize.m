% Function to return a normalized version of X where the mean value of each feature is 0 and the standard deviation is 1
% preprocessing step 
function [X_norm, mu, sigma] = featureNormalize(X) % normalizes features in X
    % variables to be returned
    X_norm = X;
    mu = zeros(1, size(X, 2));
    sigma = zeros(1, size(X, 2));   
    % Step 1: find mu
    mu = mean(X);
    % Step 2: find sigma
    sigma = std(X);
    % Step 3: find X_norm
    % update all elements of X_norm to (X_norm(i)-mu(i))/(sigma(i))
    X_norm = (X_norm-mu)./sigma;
end