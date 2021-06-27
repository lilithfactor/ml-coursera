% Function to train num_labels logistic regression classifiers and returns 
% each of these classifiers in a matrix all_theta, where the i-th row of all_theta 
% corresponds to the classifier for label i
function [all_theta] = oneVsAll(X, y, num_labels, lambda) % trains multiple logistic regression classifiers and returns all
    % the classifiers in a matrix all_theta, where the i-th row of all_theta 
    % corresponds to the classifier for label i
    m = size(X, 1);
    n = size(X, 2);
    % You need to return the following variables correctly 
    all_theta = zeros(num_labels, n + 1);
    % Add ones to the X data matrix
    X = [ones(m, 1) X];
    % setting initial theta
    theta_initial = zeros(n+1,1); 
    % setting options for fminunc
    options = optimset('GradObj','on','MaxIter',50);
    % Run fmincg to obtain the optimal theta
    for c=1:num_labels
        all_theta(c,:) = fmincg(@(t)(lrCostFunction(t, X, (y==c), lambda)), theta_initial, options);
    end
end