% lla_rpca_demo_main.m
% 2026 Kai Fukami (Tohoku University, kfukami1@tohoku.ac.jp)

% Authors:
% Pablo Koop, Isabel Scherl, and Kai Fukami
% We provide no guarantees for this code.  Use as-is and for academic research use only; no commercial use allowed without permission. For citation, please use the reference below:
%     Ref: Pablo Koop, Isabel Scherl, and Kai Fukami,
%     “Library-learning-assisted robust principal component analysis for denoising severely corrupted flow fields,”
%     in Review
%
% The code is written for educational clarity and not for speed.
% -- version 1: Aug 3, 2026


%% NACA0012 LLA-RPCA demonstration
% Compare LLA-RPCA with standard RPCA on one approximate
% vortex-shedding period of the NACA0012 dataset.

clear;
clc;
close all;

%% Main settings

active_rank = 5;
number_of_sine_modes = 500;
number_of_laplacian_modes = 500;

eta = 0.7;
rpca_lambda_scale = 1.0;

%% Locate files

script_path = fileparts(mfilename('fullpath'));

if isempty(script_path)
    script_path = pwd;
end

root_path = fullfile(script_path, '..');
functions_path = fullfile(root_path, 'functions');

addpath(functions_path);

data_path = fullfile( ...
    root_path, 'data', 'naca0012_demo_data.mat');

geometry_path = fullfile( ...
    root_path, 'data', 'naca0012_demo_geometry.inp');

%% Load data

loaded_data = load(data_path);

data = double(loaded_data.data);
data = data / loaded_data.dataset_info.data_scale;

number_of_snapshots = size(data, 1);
ny = size(data, 2);
nx = size(data, 3);

fprintf('Loaded %d snapshots.\n', number_of_snapshots);

%% Construct fluid mask

x_limits = [2, 6];
y_limits = [-1, 1];

[valid_fluid_mask, x_coordinates, y_coordinates] = ...
    build_fluid_mask( ...
        geometry_path, nx, ny, x_limits, y_limits);

%% Convert snapshots to a space-by-time matrix

data = permute(data, [2, 3, 1]);
X_clean = reshape(data, ny * nx, number_of_snapshots);

X_clean(~valid_fluid_mask(:), :) = 0;
X_clean(~isfinite(X_clean)) = 0;

%% Add sparse corruption

corruption_amplitude_multiplier = 10;
random_seed = 10;

rng(random_seed, 'twister');

X_noisy = add_signed_sparse_corruption( ...
    X_clean, valid_fluid_mask, eta, ...
    corruption_amplitude_multiplier);

%% Solver settings

solver_options.convergence_tolerance = 1e-6;
solver_options.field_change_tolerance = 1e-6;
solver_options.maximum_iterations = 500;

solver_options.nu_initial = 0.01;
solver_options.nu_growth_factor = 1.05;
solver_options.nu_maximum = 1e6;

%% Remove solid rows before decomposition

valid_rows = valid_fluid_mask(:);
X_noisy_valid = X_noisy(valid_rows, :);

rpca_lambda_0 = rpca_lambda_scale ...
    / sqrt(max(size(X_noisy_valid)));

fprintf([ ...
    'LLA-RPCA library: %d sine-product modes and %d ' ...
    'graph-Laplacian modes.\n'], ...
    number_of_sine_modes, number_of_laplacian_modes);

fprintf( ...
    'RPCA: lambda scale = %.3g, lambda_0 = %.6e.\n', ...
    rpca_lambda_scale, rpca_lambda_0);

%% LLA-RPCA

[L_lla_valid, S_lla_valid, lla_info] = ...
    lla_rpca_demo_solver( ...
        X_noisy_valid, active_rank, ...
        number_of_sine_modes, ...
        number_of_laplacian_modes, ...
        valid_fluid_mask, solver_options);

L_lla = expand_to_full_grid( ...
    L_lla_valid, valid_fluid_mask);

S_lla = expand_to_full_grid( ...
    S_lla_valid, valid_fluid_mask);

%% Standard RPCA

[L_rpca_valid, S_rpca_valid, rpca_info] = ...
    rpca_solver( ...
        X_noisy_valid, ...
        rpca_lambda_scale, ...
        solver_options.convergence_tolerance, ...
        solver_options.maximum_iterations);

L_rpca = expand_to_full_grid( ...
    L_rpca_valid, valid_fluid_mask);

S_rpca = expand_to_full_grid( ...
    S_rpca_valid, valid_fluid_mask);

%% Report results

fprintf('\nLLA-RPCA\n');
fprintf('  iterations               : %d\n', ...
    lla_info.iterations);
fprintf('  stop reason              : %s\n', ...
    lla_info.stop_reason);
fprintf('  relative primal residual : %.3e\n', ...
    lla_info.relative_primal_residual);

fprintf('\nRPCA\n');
fprintf('  iterations               : %d\n', ...
    rpca_info.iterations);
fprintf('  stop reason              : %s\n', ...
    rpca_info.stop_reason);
fprintf('  relative primal residual : %.3e\n', ...
    rpca_info.relative_primal_residual);

%% Plot first snapshot

reference_frame = reshape(X_clean(:, 1), ny, nx);
noisy_frame = reshape(X_noisy(:, 1), ny, nx);

L_lla_frame = reshape(L_lla(:, 1), ny, nx);
S_lla_frame = reshape(S_lla(:, 1), ny, nx);

L_rpca_frame = reshape(L_rpca(:, 1), ny, nx);
S_rpca_frame = reshape(S_rpca(:, 1), ny, nx);

visualization_clip_fraction = 0.75;

color_limit = visualization_clip_fraction ...
    * maximum_absolute_fluid_value( ...
        L_lla_frame, valid_fluid_mask);

figure_handle = figure( ...
    'Name', 'NACA0012 LLA-RPCA and RPCA comparison', ...
    'Color', 'w', ...
    'Units', 'normalized', ...
    'Position', [0.05, 0.10, 0.90, 0.75]);

tiledlayout(2, 3, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

plot_field(nexttile, reference_frame, valid_fluid_mask, ...
    x_coordinates, y_coordinates, color_limit, ...
    'Reference');

plot_field(nexttile, L_lla_frame, valid_fluid_mask, ...
    x_coordinates, y_coordinates, color_limit, ...
    'LLA-RPCA: L');

plot_field(nexttile, L_rpca_frame, valid_fluid_mask, ...
    x_coordinates, y_coordinates, color_limit, ...
    'RPCA: L');

plot_field(nexttile, noisy_frame, valid_fluid_mask, ...
    x_coordinates, y_coordinates, color_limit, ...
    'Observed input X');

plot_field(nexttile, S_lla_frame, valid_fluid_mask, ...
    x_coordinates, y_coordinates, color_limit, ...
    'LLA-RPCA: S');

plot_field(nexttile, S_rpca_frame, valid_fluid_mask, ...
    x_coordinates, y_coordinates, color_limit, ...
    'RPCA: S');

colormap(figure_handle, blue_white_red_colormap(256));

sgtitle(sprintf([ ...
    'NACA0012 demo: rank = %d, sine modes = %d, ' ...
    'Laplacian modes = %d, \\eta = %.2f, ' ...
    'RPCA \\lambda_0 = %.3e'], ...
    active_rank, number_of_sine_modes, ...
    number_of_laplacian_modes, eta, rpca_lambda_0));

%% Local functions

function [valid_mask, x_coordinates, y_coordinates] = ...
        build_fluid_mask( ...
            geometry_path, nx, ny, x_limits, y_limits)

    geometry = readmatrix(geometry_path, ...
        'NumHeaderLines', 1, ...
        'FileType', 'text');

    geometry = geometry(1:end-2, :);
    geometry = geometry( ...
        all(isfinite(geometry(:, 1:2)), 2), :);

    if size(geometry, 2) >= 3
        geometry_ids = geometry(:, 3);
    else
        geometry_ids = zeros(size(geometry, 1), 1);
    end

    solid_ids = unique(geometry_ids);
    solid_ids = solid_ids(solid_ids > 0);

    if isempty(solid_ids)
        solid_ids = 0;
    end

    x_coordinates = linspace( ...
        x_limits(1), x_limits(2), nx);

    y_coordinates = linspace( ...
        y_limits(1), y_limits(2), ny);

    [X_grid, Y_grid] = meshgrid( ...
        x_coordinates, y_coordinates);

    solid_mask = false(ny, nx);

    for i_object = 1:numel(solid_ids)

        object_rows = geometry_ids == solid_ids(i_object);

        polygon_x = geometry(object_rows, 1);
        polygon_y = geometry(object_rows, 2);

        solid_mask = solid_mask | inpolygon( ...
            X_grid, Y_grid, polygon_x, polygon_y);
    end

    valid_mask = ~solid_mask;
end

function X_corrupted = add_signed_sparse_corruption( ...
        X_clean, valid_fluid_mask, eta, amplitude_multiplier)

    X_corrupted = X_clean;

    valid_rows = valid_fluid_mask(:);

    valid_entry_mask = repmat( ...
        valid_rows, 1, size(X_clean, 2));

    valid_entry_indices = find(valid_entry_mask);

    number_of_corrupted_entries = round( ...
        eta * numel(valid_entry_indices));

    clean_values = X_clean(valid_rows, :);

    corruption_amplitude = amplitude_multiplier ...
        * std(clean_values(:));

    selected_entries = randperm( ...
        numel(valid_entry_indices), ...
        number_of_corrupted_entries);

    corrupted_indices = ...
        valid_entry_indices(selected_entries);

    number_of_positive_entries = floor( ...
        number_of_corrupted_entries / 2);

    X_corrupted( ...
        corrupted_indices(1:number_of_positive_entries)) = ...
        corruption_amplitude;

    X_corrupted( ...
        corrupted_indices(number_of_positive_entries + 1:end)) = ...
        -corruption_amplitude;
end

function X_full = expand_to_full_grid( ...
        X_valid, valid_fluid_mask)

    X_full = zeros( ...
        numel(valid_fluid_mask), size(X_valid, 2));

    X_full(valid_fluid_mask(:), :) = X_valid;
end

function limit = maximum_absolute_fluid_value( ...
        frame, valid_fluid_mask)

    limit = max(abs(frame(valid_fluid_mask)));

    if limit <= 0
        limit = 1;
    end
end

function plot_field( ...
        ax, frame, valid_fluid_mask, ...
        x_coordinates, y_coordinates, ...
        color_limit, title_text)

    frame(~valid_fluid_mask) = NaN;

    imagesc(ax, x_coordinates, y_coordinates, frame);

    set(ax, 'YDir', 'normal');

    axis(ax, 'image');
    axis(ax, 'tight');

    caxis(ax, [-color_limit, color_limit]);

    title(ax, title_text);
    xlabel(ax, 'x');
    ylabel(ax, 'y');

    colorbar(ax);
end

function cmap = blue_white_red_colormap(number_of_colors)

    anchors = [ ...
        0.10, 0.20, 0.70; ...
        1.00, 1.00, 1.00; ...
        0.70, 0.10, 0.10];

    anchor_positions = [0, 0.5, 1];
    query_positions = linspace(0, 1, number_of_colors);

    cmap = interp1( ...
        anchor_positions, anchors, ...
        query_positions, 'linear');
end
