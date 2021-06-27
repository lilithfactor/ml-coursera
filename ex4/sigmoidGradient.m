% Function to compute the gradient of the sigmoid function
% evaluated at z. This should work regardless if z is a matrix or a
% vector. In particular, if z is a vector or matrix, you should return
% the gradient for each element
function g = sigmoidGradient(z) % returns the gradient of the sigmoid function
    % evaluated at z
    %   g = SIGMOIDGRADIENT(z) .
    g = zeros(size(z));
    g = sigmoid(z).*(1-sigmoid(z));
end