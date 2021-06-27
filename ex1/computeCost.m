% Function to Compute cost of using theta as the parameter for linear regression to fit the data points in X and y
function J = computeCost(X, y, theta)
    % number of training examples
    m = length(y); 
    % predicting hypothesis
    prediction = X*theta;
    % error
    error = (prediction-y).^2;
    % cost of theta for Linear Regression
    J = 1/(2*m)*sum(error);
end