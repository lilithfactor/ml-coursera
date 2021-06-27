% Function to return the cost and gradient for the
% collaborative filtering problem
function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda) % Collaborative filtering cost function
    % Unfold the U and W matrices from params
    X = reshape(params(1:num_movies*num_features), num_movies, num_features);
    Theta = reshape(params(num_movies*num_features+1:end), ...
                    num_users, num_features);
    % You need to return the following values correctly
    J = 0;
    X_grad = zeros(size(X));
    Theta_grad = zeros(size(Theta));
    % no regularization
    error = (X*Theta')-Y;

    J = (1/2)*sum(sum(error.^2.*R));

    X_grad = (error.*R)*Theta; % dim: nm x n
    Theta_grad = (error.*R)'*X; % dim: nu x n

    % with regularization
    reg_theta = (lambda/2)*sum(sum(Theta.^2));
    reg_x = (lambda/2)*sum(sum(X.^2));
    % adding regularization to cost function
    J = J + reg_theta + reg_x;

    X_grad = X_grad + lambda*X; % dim: nm xn
    Theta_grad = Theta_grad + lambda*Theta; % dim: nu x n

    grad = [X_grad(:); Theta_grad(:)];
end