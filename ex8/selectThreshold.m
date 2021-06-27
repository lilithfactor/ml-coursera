% Function to find the best threshold to use for selecting outliers based on the results from a
% validation set (pval) and the ground truth (yval)
function [bestEpsilon bestF1] = selectThreshold(yval, pval) % Find the best 
    % threshold (epsilon) to use for selecting outliers
    bestEpsilon = 0;
    bestF1 = 0;
    F1 = 0;
    
    stepsize = (max(pval) - min(pval)) / 1000;
    % selecting the best epsilon based on F1 score and storing in
    % bestEpsilon and bestF1
    for epsilon = min(pval):stepsize:max(pval)
        % selecting values in pval with prob less than current epsilon
        cvPredictions = (pval < epsilon); % m x 1 
        % calculating true positive, false positive, false negative
        tp = sum((cvPredictions == 1) & (yval == 1)); % m x 1
        fp = sum((cvPredictions == 1) & (yval == 0)); % m x 1
        fn = sum((cvPredictions == 0) & (yval == 1)); % m x 1
        % calculating precision and recall
        prec = tp/(tp+fp); 
        rec = tp/(tp+fn);

        F1 = 2*prec*rec / (prec + rec);

        if F1 > bestF1
           bestF1 = F1;
           bestEpsilon = epsilon;
        end
    end
end