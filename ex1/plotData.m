% Function to plot the data points and gives the figure axes labels of population and profit.
function plotData(x, y) % Plots the data points x and y into a new figure
    % to open a new figure window
    figure; 
    % plot data using red crosses
    plot(x, y, 'rx', 'MarkerSize', 10);
    xlabel('Population [10000s]');
    ylabel('Revenue [$10000s]');
    title('plotData.m');
end