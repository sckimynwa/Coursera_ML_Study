%%=================== Part 1: Get Data ===================

fprintf('Makie Datasets\n');
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y);
fprintf('Plot Finished\n');

%%=================== Part 2: define Functions ===========

theta = zeros(2, 1);
costX = [ones(m, 1), data(:, 1)];
iterations = 1500;
alpha = 0.01;

function J = computeCost(X, y, theta)
    J = 0;
    m = length(y);
    costX = [ones(m, 1), X(:, 1)];
    h = costX * theta;
    J = (1/(2*m)) * sum((h-y).^2);
end

fprintf('Visualizing J\n');

%%==================== Part 3: Visualizing ================

theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

J_vals = zeros(length(theta0_vals), length(theta1_vals));

for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
        t = [theta0_vals(i); theta1_vals(j)];
        J_vals(i, j) = computeCost(X, y, t);
    end
end

J_vals = J_vals';
figure;
surf(theta0_vals, theta1_vals, J_vals);
xlabel('\theta_0');
ylabel('\theta_1');