% Function to compute the projection of 
% the normalized inputs X into the reduced dimensional space spanned by
% the first K columns of U
function Z = projectData(X, U, K) % Computes the reduced data representation when projecting only 
    % on to the top k eigenvectors
    Z = zeros(size(X, 1), K);
    % n x K
    U_reduce = U(:,[1:K]);   
    % m x k
    Z = X * U_reduce;        
end