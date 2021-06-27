% Function to display the progress of k-Means as it is running
function plotProgresskMeans(X, centroids, previous, idx, K, i) % plots the data
    % points with colors assigned to each centroid

    % Plot the examples
    plotDataPoints(X, idx, K);

    % Plot the centroids as black x's
    plot(centroids(:,1), centroids(:,2), 'x', ...
         'MarkerEdgeColor','k', ...
         'MarkerSize', 10, 'LineWidth', 3);

    % Plot the history of the centroids with lines
    for j=1:size(centroids,1)
        drawLine(centroids(j, :), previous(j, :));
    end

    % Title
    title(sprintf('Iteration number %d', i));
end