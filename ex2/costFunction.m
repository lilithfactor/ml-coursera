% Function to compute cost of using theta as parameter for logistic regression and the gradient of the cost w.r.t. to the parameters.
function [J, grad] = costFunction(theta, X, y) % Compute cost and gradient for logistic regression
    % number of training examples
    m = length(y); 
    % variables to be returned
    J = 0;
    grad = zeros(size(theta));
    % to be passed to sigmoid function
    z = X*theta; % dim: mx1
    % hypothesis function h
    h = sigmoid(z); % dim: mx1
    % vectorized cost function for Logistic Regression
    J = (1/m)*( (-y'*log(h)) - ((1-y)'*log(1-h)) ); % scalar 
    % gradient
    grad = (1/m)*(X')*(h-y);
end