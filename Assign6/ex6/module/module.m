% initialize
clear; close all; clc;

% Plot Data Function
function plotData(X, y)
    pos = find(y == 1);
    neg = find(y == 0);

    % plot examples
    plot(X(pos, 1), X(pos, 2), 'k+', 'LineWidth', 1, 'MarkerSize', 7);
    hold on;
    plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
    hold off;
end

load('ex6data1.mat');
plotData(X, y);

pause;

% Training Linear SVM
C = 1;

