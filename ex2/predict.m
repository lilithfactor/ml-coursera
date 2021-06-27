% Function to compute the predictions for X using a threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
function p = predict(theta, X) % Predict whether the label is 0 or 1 using learned logistic regression parameters theta
    % Number of training examples
    m = size(X, 1); 
    % variables to be returned 
    p = zeros(m, 1);
    h = sigmoid(X*theta);
    p=(h>=0.5);
end