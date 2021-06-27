% Function to compute the sigmoid of z
function g = sigmoid(z) % Compute sigmoid function
    % J = sigmoid(z) 
    g = 1.0 ./ (1.0+exp(-z));
end