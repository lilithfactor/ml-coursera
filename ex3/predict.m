% Function to output the predicted label of X given the trained weights of a neural network (Theta1, Theta2)
function p = predict(Theta1, Theta2, X) % Predict the label of an input given a trained neural network
    m = size(X, 1);
    num_labels = size(Theta2, 1);
    % variables to be returned
    p = zeros(size(X, 1), 1);
    % 5000 x 401 == no_of_input_images x no_of_features 
    % Adding 1 in X
    a1 = [ones(m,1) X];  
    % No. of rows = no. of input images
    % No. of Column = No. of features in each image
    z2 = a1 * Theta1';  % dim: 5000 x 25
    a2 = sigmoid(z2);   % dim: 5000 x 25
    a2 =  [ones(size(a2,1),1) a2];  % dim: 5000 x 26
    z3 = a2 * Theta2';  % dim: 5000 x 10
    a3 = sigmoid(z3);  % dim 5000 x 10
    % returns maximum element in each row, max. probability and its index for each input image
    % p: predicted output (index)
    % prob: probability of predicted output
    [prob, p] = max(a3,[],2); 
end