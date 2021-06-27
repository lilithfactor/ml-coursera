% Function to recover an approximation the 
% original data that has been reduced to K dimensions
function X_rec = recoverData(Z, U, K) % Recovers an approximation of the original data when using the 
    % projected data
    X_rec = zeros(size(Z, 1), size(U, 1));

    % n x k
    U_reduce = U(:,1:K);   
    % m x n
    X_rec = Z * U_reduce'; 
end