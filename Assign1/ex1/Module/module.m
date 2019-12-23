%% Module 1 for Practice & review
clear; close all; clc

%% getData Function (single, multi)
function [X, y] = getSingleData()
    fprintf('Get Single Data');
    data = load('ex1data1.txt');
    X = data(:, 1);
    y = data(:, 2);
end

function [X, y] = getMultiData()
    fprintf('Get Multi Data');
    data = load('ex1data2.txt');
    X = data(:, 1:2);
    y = data(:, 3);
end


%% Plotting data Function (2D space) - Single
function plotSingleData(x, y)
    plot(x, y, 'ro', 'MarkerSize', 5);
    xlabel("this is label x");
    ylabel("this is label y");
end

%% Calculate CostFunction - single
%% must check vector format (e.g h = 97*1, x = 97*2, theta = 2*1)
function J = costFunction(x, y, theta)
    m = length(y);
    h = x * theta; 
    J = (1/(2*m)) * sum((h-y).^2);
end

%% 3D visualization of Cost Function
%% Independent Function (no need to set params)
function visualizeCost()
    [X, y] = getSingleData();
    m = length(y);
    x = [ones(m, 1), X(:, 1)];
    theta0_vals = linspace(-10, 10, 100);
    theta1_vals = linspace(-1, 4, 100);
    costJ = zeros(length(theta0_vals), length(theta1_vals));

    % Fill out
    for i = 1: length(theta0_vals)
        for j = 1: length(theta1_vals)
            t = [theta0_vals(i), theta1_vals(j)];
            costJ(i, j) = costFunction(x, y, t');
            %% t' because of format
        end
    end

    % Surface
    figure;
    surf(theta0_vals, theta1_vals, costJ);
    xlabel('theta0');
    ylabel('theta1');
end

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
    m = length(y);
    J_history = zeros(num_iters, 1);

    % Iterations
    for iter = 1: num_iters
        h = X * theta;
        step = (1/m) * alpha * sum((h-y)*.X)';
        theta -= step;

        J_history(iter) = costFunction(X, y, theta);
    end
end

%% Test Function
