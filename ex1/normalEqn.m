% Function to compute the closed-form solution to linear regression using the normal equations
function [theta] = normalEqn(X, y) % Computes the closed-form solution to linear regression
    theta = zeros(size(X, 2), 1);
    % normal equation
    theta = pinv(X'*X)*X'*y
end
