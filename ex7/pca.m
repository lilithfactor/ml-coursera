% Function to compute eigenvectors of the covariance matrix of X
% Returns the eigenvectors U, the eigenvalues (on diagonal) in S
function [U, S] = pca(X) % Run principal component analysis on the dataset X
    [m, n] = size(X);
    % variables to be returned
    U = zeros(n);
    S = zeros(n); 
    Sigma = (1/m)*(X'*X); % dim: nxn
    [U, S, V] = svd(Sigma);
end