% Function to compute the cost of using theta as the parameter for linear regression 
% to fit the data points in X and y    
function [J, grad] = linearRegCostFunction(X, y, theta, lambda) % Compute cost and gradient for 
    % regularized linear regression with multiple variables
    
    % number of training examples
    m = length(y); 
    % variables to be returned
    J = 0;
    grad = zeros(size(theta));
    
    % calculating regularization Term
    reg = (lambda/(2*m))*sum(theta(2:end).^2);
    
    % calculating hypothesis
    hx = X*theta; %12x1
    
    % calculating regularized J
    J = (1/(2*m))*sum((hx-y).^2) + reg; % number

    % calculating gradient
    grad(1) = (1/m)*(X(:, 1)'*(hx-y)); %1x1 Number
    % regularized gradient for 2:end
    grad(2:end) = (1/m)*(X(:, 2:end)'*(hx-y)) + (lambda/m)*theta(2:end);
    % unroll gradient into column vector
    grad = grad(:);
end