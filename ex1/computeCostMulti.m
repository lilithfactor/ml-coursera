% Function to compute cost of using theta as a parameter for linear
% regression to fit the data points in X and y (with multiple variables)
function J = computeCostMulti(X, y, theta)
    % number of training examples
    m = length(y); 
    % variable to be returned
    J = 0;
    % predicting hypothesis of all m
    prediction = X*theta;
    % error
    error = (prediction-y).^2;
    % cost of theta for linear regression
    J = 1/(2*m)*sum(error);
end