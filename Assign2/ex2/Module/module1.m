%% Module for Assignment 2
% Logistic Regression

%% Initialize
clear; close all; clc

%% Plot Data
function plotData()
    % Load data
    data = load('ex2data1.txt');
    X = data(:, [1,2]);
    y = data(:, 3);

    % Get data & Plot
    pos = find(y==1);
    neg = find(y==0);

    figure; hold on;
    plot(X(pos, 1), X(pos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 5);
    plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'b', 'LineWidth', 2, 'MarkerSize', 5);
    
    xlabel('Exam1 score');
    ylabel('Exam2 score');
    legend('Admitted', 'Not Admitted');
    % hold off;
end

function g = sigmoid(z)
    % initial value set
    g = zeros(size(z));
    g = 1./(1+exp(-1*z));
end

function [J, grad] = costFunction(theta, X, y)
    % initial value set
    m = length(y);
    J = 0;
    grad = zeros(size(theta));

    % Calculate
    J = (-1/m) * sum(y.*log(sigmoid(X*theta)) + (1-y).*log(1-sigmoid(X*theta)));
    step = sigmoid(X*theta) - y;
    grad = (1/m) * (X' * step);
end

function [cost, grad] = optimWithFminunc()
    % initial value set
    % Load data
    data = load('ex2data1.txt');
    X = data(:, [1,2]);
    y = data(:, 3);
    [m, n] = size(X);
    X = [ones(m, 1), X];
    initial_theta = zeros(n+1, 1);
    options = optimset('GradObj', 'on', 'MaxIter', 400);

    % Using fminunc
    [cost, grad] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
end

function plotDecisionBoundary()
    % initial value set
    [cost, grad] = optimWithFminunc()
    data = load('ex2data1.txt');
    X = data(:, [1,2]);
    y = data(:, 3);
    [m, n] = size(X);
    X = [ones(m, 1), X];
    theta = cost;

    % plotDecision Boundary
    plotData();
    hold on;

    if size(X, 2) <= 3
        % Only need 2 points to define a line, so choose two endpoints
        plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
    
        % Calculate the decision boundary line
        plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
    
        % Plot, and adjust axes for better viewing
        plot(plot_x, plot_y)
        
        % Legend, specific for the exercise
        legend('Admitted', 'Not admitted', 'Decision Boundary')
        axis([30, 100, 30, 100])
    else
        % Here is the grid range
        u = linspace(-1, 1.5, 50);
        v = linspace(-1, 1.5, 50);
    
        z = zeros(length(u), length(v));
        % Evaluate z = theta*x over the grid
        for i = 1:length(u)
            for j = 1:length(v)
                z(i,j) = mapFeature(u(i), v(j))*theta;
            end
        end
        z = z'; % important to transpose z before calling contour
    
        % Plot z = 0
        % Notice you need to specify the range [0, 0]
        contour(u, v, z, [0, 0], 'LineWidth', 2)
    end
    hold off
end

function [p, accuracy] = predict()
    % initialize
    [cost, grad] = optimWithFminunc();
    data = load('ex2data1.txt');
    X = data(:, [1,2]);
    y = data(:, 3);
    m = size(X, 1);
    p = zeros(m, 1);
    X = [ones(m, 1), X];

    % predict
    p = round(sigmoid(X * cost));

    % calculate accuracy
    accuracy = mean(double(p==y)) * 100;

end

%% Test Function 
[p, accuracy] = predict();
