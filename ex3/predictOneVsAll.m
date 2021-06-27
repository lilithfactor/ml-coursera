% will return a vector of predictions
% for each example in the matrix X. Note that X contains the examples in
% rows. all_theta is a matrix where the i-th row is a trained logistic
% regression theta vector for the i-th class. You should set p to a vector
% of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
% for 4 examples) 
function p = predictOneVsAll(all_theta, X) % Predict the label for a trained one-vs-all classifier. The labels 
    % are in the range 1..K, where K = size(all_theta, 1)
    m = size(X, 1);
    num_labels = size(all_theta, 1);
    % variables to be returned
    p = zeros(size(X, 1), 1);
    % Add ones to the X data matrix
    X = [ones(m, 1) X];
    % num_labels = No. of output classifier (Here, it is 10)
    % all_theta dim = 10 x 401 = num_labels x (input_layer_size+1) == num_labels x (no_of_features+1)
    % returns maximum element in each row  == max. probability and its index for each input image
    % p: predicted output (index)
    % prob: probability of predicted output
    prob_mat = X * all_theta'; % dim: 5000 x 10 == no_of_input_image x num_labels
    [prob, p] = max(prob_mat,[],2); % dim: m x 1 
end