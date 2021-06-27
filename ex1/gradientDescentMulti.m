% function to update theta by taking num_iters gradient steps with learning rate alpha
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters) % Performs gradient descent to learn theta
    % number of training examples
    m = length(y); 
    J_history = zeros(num_iters, 1);
    for iter = 1:num_iters
        % error term in theta update
        error = (X*theta) - y;
        % theta update
        theta = theta - alpha*(1/m)*(X'*error); 
        % adding cost J after every iteration    
        J_history(iter) = computeCostMulti(X, y, theta);
    end
end