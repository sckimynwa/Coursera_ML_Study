%% Initialization
clear; close all; clc;

%% Loading and Visualizing Data
load('ex5data1.mat');

m = size(X, 1);
%% Part 1 : Plot Traning Data
plot(X, y, 'rx', 'MarkerSize', 5, 'LineWidth', 1.5);
xlabel('x axis');
ylabel('y axis');
pause;

close all; clc;

%% Part 2: Regularized Linear Cost Function
function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
    m = length(y);
    J = 0;
    grad = zeros(size(theta));

    temp = theta;
    temp(1) = 0;

    J = (1/(2*m)) * sum(((X*theta)-y).^2) + (lambda/(2*m)) * sum(temp.^2);
    err = X*theta - y;
    grad = (1/m) * (X'*err) + (lambda/m)*temp;
    grad = grad(:);
end

%% Regularized Linear Regression Cost
theta = [1 ; 1];
J = linearRegCostFunction([ones(m, 1), X], y, theta, 1);

%% Part 3 : Train Linear Regression

function [theta] = trainLinearReg(X, y, lambda)
    init_theta = zeros(size(X, 2), 1);
    costFunction = @(t) linearRegCostFunction(X, y, t, lambda);
    options = optimset('MaxIter', 200, 'GradObj', 'on');
    theta = fmincg(costFunction, init_theta, options);
end

lambda = 0;
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);

%% plot data
plot(X, y, 'rx', 'MarkerSize', 3, 'LineWidth', 1.5);
xlabel('x-axis');
ylabel('y-axis');
hold on;
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2);
hold off;
pause;

close all; clc;

%% Part 4 : Learning Curve
function [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda)
    m = size(X, 1);
    error_train = zeros(m, 1);
    error_val = zeros(m, 1);

    for i = 1:m
        Xtrain = X(1:i, :);
        Ytrain = y(1:i);
        %% Add data for traning -> train theta
        theta = trainLinearReg(Xtrain, Ytrain, lambda);

        error_train(i) = linearRegCostFunction(Xtrain, Ytrain, theta, 0);
        error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);

    end

end

[error_train, error_val] = learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, lambda);

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

close all; clc;

%% Part 5 : Validation Curve
function [lambda_vec, error_train, error_val] = validationCurve(X, y, Xval, yval)
    lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
    error_train = zeros(length(lambda_vec), 1);
    error_val = zeros(length(lambda_vec), 1);
    m = length(lambda_vec)

    for i = 1:m
        lambda = lamdba_vec(i);
        theta = trainLinearReg(X, y, lambda);
        error_train(i) = linearRegCostFunction(X, y, theta, 0);
        error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
    end
end
