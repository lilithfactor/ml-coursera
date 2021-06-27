% Function to update theta by taking num_iters gradient steps with learning rate alpha
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters) % performs gradient descent to learn theta
    % number of training examples
    m = length(y); 
    % initializing var to hold values of J
    J_history = zeros(num_iters, 1);
    % updating theta num_iters times
    for iter = 1:num_iters
        % error term in theta update
        error = (X * theta) - y;
        % updating theta
        theta = theta - alpha*(1/m)*X'*error;
        % add cost function after every iteration using computeCost()    
        J_history(iter) = computeCost(X, y, theta);
    end
end