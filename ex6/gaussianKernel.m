% Function to return a gaussian kernel between x1 and x2
% and returns the value in sim
function sim = gaussianKernel(x1, x2, sigma) % returns a radial basis function kernel between x1 and x2
    % Ensure that x1 and x2 are column vectors
    x1 = x1(:); 
    x2 = x2(:);
    % variable to be returned
    sim = 0;
    sim = exp(-1*sum(abs(x1-x2).^2)/(2*sigma^2));
end