% Function to compute the sigmoid of z
function g = sigmoid(z) % Compute sigmoid function
    %   g = SIGMOID(z)
    g = zeros(size(z));
    g = (1+exp(-z)).^(-1);
end