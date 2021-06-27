% Function to return your choice of C and sigma for Part 3 of the exercise
% where you select the optimal (C, sigma) learning parameters to use for SVM
% with RBF kernel
function [C, sigma] = dataset3Params(X, y, Xval, yval) % Function to return your choice of C and sigma

    % variables to be returned
    C = 1;
    sigma = 0.3;
    
    Cvals = [0.01 0.03 0.1 0.3 1 3 10 30]';
    sigmavals = [0.01 0.03 0.1 0.3 1 3 10 30]';

    % to store prediction error
    predictionError = zeros(length(Cvals), length(sigmavals));
    % to store predictionErrors and corresponsing C and sigma values
    result = zeros(length(Cvals) + length(sigmavals), 3);
    % variable to iterate results of all models
    row = 1;

    for i = 1:length(Cvals)
        for j = 1:length(sigmavals)
            % choosing pairs of C and sigma from list for a model
            Ctemp = Cvals(i);
            sigmatemp = sigmavals(j);
            % training SVM for every model
            model = svmTrain(X, y, Ctemp, @(x1, x2)gaussianKernel(x1, x2, sigmatemp));
            % predicting labels and returning predictions on Cross Validation Set
            predictions = svmPredict(model, Xval);
            % computing prediction error 
            predictionError(i, j) = mean(double(predictions ~= yval));
            % storing predictionError and corresponding C and sigma
            result(row, :) = [predictionError(i, j), Ctemp, sigmatemp];
            % updating row to store next result
            row = row + 1;
        end
    end

    % sorting predictionError in ascending order to get least error
    sortedResult = sortrows(result, 1);
    % getting C and sigma corresponsing to least predictionError
    C = sortedResult(1, 2);
    sigma = sortedResult(1, 3);
end