% Function to map the two input features to quadratic features used in the regularization exercise
% Returns a new feature array with more features, comprising of 
% X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
function output = mapFeature(X1, X2) % Feature mapping function to polynomial features
    degree = 6;
    output = ones(size(X1(:,1)));
    for i = 1:degree
        for j = 0:i
            output(:, end+1) = (X1.^(i-j)).*(X2.^j);
        end
    end
end