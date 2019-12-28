%% Module 3

%% Initialize
clear; close all; clc

%% displaydata function
function [h, display_array] = displayData(X, example_width)
    % set example_width automatically if not set
    if ~exist('example_width', 'var') || isempty(example_width)
        example_width = round(sqrt(size(X, 2)));
    end

    % Assume that image is square (row == col)

    % Gray Image
    colormap(gray);

    % Compute rows, cols
    [m n] = size(X);
    example_height = (n / example_width);

    % Compute number of items to display (10 * 10)
    display_rows = floor(sqrt(m));
    display_cols = ceil(m / display_rows);

    % between images padding
    pad = 1;

    % Setup blank display
    display_array = - ones(pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad));

    % copy each example into a patch on the display array
    curr_ex = 1;
    for j = 1:display_rows
        for i = 1:display_cols
            if curr_ex > m,
                break;
            end

            max_val = max(abs(X(curr_ex, :))); % scale from -1 to 1;
            display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
            curr_ex = curr_ex + 1;
        end
        if curr_ex > m, 
            break; 
        end
    end

    % display image
    h = imagesc(display_array, [-1 1]);

    % Do not show axis
    axis image off

    drawnow;
end

%% Test

% Plot data sample (10*10 matrix)
clear; close all; clc;
load('ex3data1.mat');
m = size(X, 1); % 5000;
rand_indices = randperm(m);
fprintf("%d\n", m);
sel = X(rand_indices(1:100), :);

displayData(sel);

pause;