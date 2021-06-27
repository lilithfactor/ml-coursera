% Function to output the predicted label of X given the
% trained weights of a neural network (Theta1, Theta2)
function p = predict(Theta1, Theta2, X) % Predict the label of an input given a trained neural network

    m = size(X, 1);
    num_labels = size(Theta2, 1);

    % variables to be returned
    p = zeros(size(X, 1), 1);

    h1 = sigmoid([ones(m, 1) X]*Theta1');
    h2 = sigmoid([ones(m, 1) h1]*Theta2');
    [~, p] = max(h2, [], 2);
end
