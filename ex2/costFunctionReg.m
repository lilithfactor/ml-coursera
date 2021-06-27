% Function to compute the cost of using theta as the parameter for regularized logistic regression and the gradient of the cost w.r.t. to the parameters
function [J, grad] = costFunctionReg(theta, X, y, lambda) % Compute cost and gradient for logistic regression with regularization
    % number of training examples
    m = length(y); 

    % variables to be returned
    J = 0;
    grad = zeros(size(theta));
    
    %%%Dimensions%%%
    % theta dim: (n+1)x1
    % X dim: mx(n+1)
    % y dim: mx1
    % grad dim: (n+1)x1

    z = X*theta; % dim: mx1
    h = sigmoid(z); % dim: mx1
    
    % from theta1 to thetan (excluding theta0)
    rg = (lambda/(2*m))*sum(theta(2:end).^2); % scalar
    
    % regularized Cost Function
    J = (1/m)*sum( (-y.*log(h)) - ((1-y).*log(1-h)) ) + rg; % scalar
    grad(1) = (1/m)*(X(:, 1)')*(h-y); % dim: 1x1
    % regularized Gradient
    grad(2:end) = (1/m)*(X(:, 2:end)')*(h-y) + (lambda/m)*theta(2:end); % dim: nx1
end