% Function to return the train and
% cross validation set errors for a learning curve
function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda) % Generates the train and cross 
    % validation set errors needed to plot a learning curve
    
    % Number of training examples
    m = size(X, 1);
    % variables to be returned
    error_train = zeros(m, 1); % dim: mx1
    error_val   = zeros(m, 1); % dim: mx1

    for i = 1:m
        % collecting training examples
        X_train = X(1:i, :);
        y_train = y(1:i);
        % collecting optimized Theta values for X_train using fmincg in trainLinearReg()
        theta = trainLinearReg(X_train, y_train, lambda); % lambda = 0
        % collecting errors for Xtrain, ytrain in (i:i) and for complete Xval, y val
        error_train(i) = linearRegCostFunction(X_train, y_train, theta, 0); % lambda = 0
        error_val(i) = linearRegCostFunction(Xval, yval, theta, 0); % lambda = 0
    end
end