% Function to plot the data points with + for the positive examples
% and o for the negative examples. X is assumed to be a Mx2 matrix
function plotData(X, y) % Plots the data points X and y into a new figure 
    % Create New Figure
    figure; 
    % to add multiple plots on the same figure
    hold on;
    pos = find(y==1); % storing indices of positive examples
    neg = find(y==0); % storing indices of negative examples
    % plotting positive examples as '+'
    plot(X(pos,1), X(pos,2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
    % plotting negative examples as 'O'
    plot(X(neg,1), X(neg,2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
    hold off;
end