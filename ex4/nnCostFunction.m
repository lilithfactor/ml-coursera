% Function to computes the cost and gradient of the neural network. The
% parameters for the neural network are "unrolled" into the vector
% nn_params and need to be converted back into the weight matrices
function [J, grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,...                                    
                                   num_labels, X, y, lambda) % Implements the neural network 
    % cost function for a two layer neural network which performs classification
    
    % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    % for our 2 layer neural network
    Theta1 = reshape(nn_params(1:hidden_layer_size*(input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));% 25 x 401
    Theta2 = reshape(nn_params((1 + (hidden_layer_size*(input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1)); % 10 x 26

    m = size(X, 1);

    % variables to be returned
    J = 0;
    Theta1_grad = zeros(size(Theta1)); % dim: 25 x 401
    Theta2_grad = zeros(size(Theta2)); % dim: 10 x 26
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Part 1: Feedforward NN, Return J %%%
    % Calculate J without regularization %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % adding column of ones to X
    X = [ones(m,1), X];
    % activation for first layer is input layer + bias unit
    a1 = X; % 5000 x 401
    % calculating z2
    z2 = a1*Theta1'; % m x hidden_layer_size == 5000 x 25
    % calculating a2
    a2 = sigmoid(z2); % m x hidden_layer_size == 5000 x 25
    % adding column of 1's for Bias Unit
    a2 = [ones(size(a2,1),1), a2]; % mx(hidden_layer_size+1)=5000x26
    % calculating z3
    z3 = a2*Theta2'; % m x num_labels == 5000 x 10
    % calculating a3
    a3 = sigmoid(z3); % m x num_labels == 5000 x 10
    % calculating hx
    hx = a3; % m x num_labels == 5000 x 10
    % converting y into vector of 0's and 1's for classification
    yVec = (1:num_labels)==y; % m x num_labels == 5000 x 10
    % Calculating Cost Function without Regularization
    J = (1/m)*sum(sum((-yVec.*log(hx)) - ((1-yVec).*log(1-hx)))); % number

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Part 2: Implement Back Prop %%%%%%%%%%%
    % Check Part 2 By running checkNNGradients %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % activation for first layer is input layer + bias unit
    A1 = X; % 5000 x 401
    % calculating Z2
    Z2 = A1*Theta1'; % m x hidden_layer_size == 5000 x 25
    % calculating A2
    A2 = sigmoid(Z2); % m x hidden_layer_size == 5000 x 25
    % adding column of 1's for Bias Unit
    A2 = [ones(size(A2,1),1), A2]; % mx(hidden_layer_size+1)=5000x26
    % calculating Z3
    Z3 = A2*Theta2'; % m x num_labels == 5000 x 10
    % calculating a3
    A3 = sigmoid(Z3); % m x num_labels == 5000 x 10
    % h_x = a3; % m x num_labels == 5000 x 10
    yVec = (1:num_labels)==y; % m x num_labels == 5000 x 10
    % Calculating D3, D2
    % calculating delta(error) for Layer 3 (last layer)
    D3 = A3 - yVec; % 5000x10
    % calculating delta(error) for Layer 2 (Second Last)
    D2 = (D3*Theta2).*[ones(size(Z2,1), 1) sigmoidGradient(Z2)]; % 5000x26
    % delta need not be calculated for the first/input layer
    % removing bias unit for Delta 2
    D2 = D2(:,2:end); % dim: 5000x25
    % Calculating Theta1/2_gradient 
    Theta1_grad = (1/m)*(D2'*A1); % dim: 25x401
    Theta2_grad = (1/m)*(D3'*A2); % dim: 10x26

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%% Part 3: Regionalization %%%%%%%%%%%
    % Add Regionalization term to J and grad %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % using regionalization for all theta 
    reg = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+ ...
        sum(sum(Theta2(:,2:end).^2))); % number
    % regularization of Cost Function
    J = J + reg; % number
    % regularization of Gradients
    Theta1_grad = Theta1_grad + ...
        (lambda/m)*[zeros(size(Theta1,1),1) Theta1(:,2:end)]; %25x401
    Theta2_grad = Theta2_grad + ...
        (lambda/m)*[zeros(size(Theta2,1),1) Theta2(:,2:end)]; %10x26

    % Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];
end