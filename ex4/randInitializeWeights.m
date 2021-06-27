% Function to randomly initialize the weights 
% of a layer with L_in incoming connections and L_out outgoing 
% connections
function W = randInitializeWeights(L_in, L_out) % Randomly initialize the weights 
    % of a layer with L_in incoming connections and L_out outgoing connections

    % variables to be returned
    W = zeros(L_out, 1 + L_in);
    epsilon = 0.12;
    W = rand(L_out, 1+L_in)*2*epsilon-epsilon;
end