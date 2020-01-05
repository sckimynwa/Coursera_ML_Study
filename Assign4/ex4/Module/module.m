%% initialization
clear; close all; clc;

%% Setup the parameters
input_layer_size = 400;
hidden_layer_size = 25;
num_labels = 10;

%% Loading & Visualizing the Data.
load('ex4data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

%% Display Function
function [h, display_array] = displayData(X, example_width)
    if ~exist('example_width', 'var') || isempty(example_width)
        example_width = round(sqrt(size(X, 2)));
    end

    colormap(gray);
    [m, n] = size(X);
    example_height = (n / example_width);

    % Compute number of items so display
    display_rows = floor(sqrt(m));
    display_cols = ceil(m/display_rows);

    pad = 1;

    display_array = -ones(pad+display_rows*(example_height+pad), ...
    pad+display_cols*(example_width+pad));

    % Color the array
    cur = 1;
    for j = 1:display_rows
        for i = 1:display_cols
            if cur > m,
                break;
            end
            max_val = max(abs(X(curr_ex, :)));
            display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
                        pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
                            reshape(X(curr_ex, :), example_height, example_width) / max_val;
            curr_ex = curr_ex + 1;
        end
        if cur > m,
            break;
        end
    end

    h = imagesc(display_array, [-1,1]);
    axis image off
    drawnow;
end
pause;

%% Loading Parameters
clear; close all; clc;

load('ex4weights.mat');
%% Parameters Unrolling - needed to be reshaped.
nn_params = [Theta1(:) ; Theta2(:)]

% Sigmoid Functions
function g = sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
end

% Sigmoid Gradient
function g = sigmoidGradient(z)
    g = zeros(size(z));
    g = sigmoid(z) .* (1-sigmoid(z));
end

% Random Initialize Weights
function W = randinitializeWeights(L_in, L_out)
    W = zeros(L_out, 1+L_in);
    epsilon_init = 0.12;
    W = rand(L_out, 1+L_in) * 2 * epsilon_init - epsilon_init;
end

%% Forward Propagation
lambda = 0;

J = nnCostFunction(nn_params, input_layer_size, ...
hidden_layer_size, num_labels, X, y, lambda);

function [J, grad] = nnCostFunction(nn_params, input_layer_size, ...
    hidden_layer_size, num_labels, X, y, lambda)

    % reshape unrolled parameters to theta
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1+(hidden_layer_size * (input_layer_size+1))):end), ...
    num_labels, (hidden_layer_size+1));

    % initialize
    m = size(X, 1); % number of test cases
    J = 0;
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));

    % implementation

    %% Part 1 - forward propagation
    a1 = [ones(m, 1), X];
    z2 = a1 * Theta1';
    a2 = sigmoid(z2);

    a2 = [ones(m, 1), a2];
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);

    h = a3; %% result of forward propagation

    %% change in to binary form 0000100000 using repmat
    %% calculate cost function
    y = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
    J = (-1/m) * sum(sum(y.log(h) + (1-y).*log(1-h)));

    %% Regularization
    regTheta1 = Theta1(:, 2:end);
    regTheta2 = Theta2(:, 2:end);
    regVal = (lambda/(2*m)) * sum(sum(regTheta1.^2) + sum(regTheta2.^2));

    J += regVal;

    %% Part 2 - backpropagation
    delta1 = zeros(size(Theta1));
    delta2 = zeros(size(Theta2));

    for t = 1:m %% t means traning sets

        % set t-th traning example
        a1t = a1(t, :);
        a2t = a2(t, :);
        a3t = a3(t, :);
        yt = y(t, :);

        % back-propagate
        d3 = (a3t-yt);
        d2 = Theta2'*d3' .* sigmoidGradient([1;Theta1 * a1t']);

        delta1 += d2(2:end)*a1t;
        delta2 += d3' * a2t;
        
    end

    Theta1_grad = 1/m * delta1 + (lambda/m) * [zeros(size(Theta1, 1), 1) regTheta1];
    Theta2_grad = 1/m * delta2 + (lambda/m) * [zeros(size(Theta2, 1), 1) regTheta2];

    % Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];

end