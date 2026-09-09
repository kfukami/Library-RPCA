% main_llrpca.m
% 2026 Kai Fukami (Tohoku University, kfukami1@tohoku.ac.jp)

%% Authors:
% Pablo Koop, Isabel Scherl, and Kai Fukami
% We provide no guarantees for this code.  Use as-is and for academic research use only; no commercial use allowed without permission. For citation, please use the reference below:
%     Ref: Pablo Koop, Isabel Scherl, and Kai Fukami,
%     “Library-learning-assisted robust principal component analysis for denoising severely corrupted flow fields,”
%     in Review
%
% The code is written for educational clarity and not for speed.
% -- version 1: Sep 9, 2026


clear; clc; close all;
set(groot, 'defaultFigureVisible', 'on');

% 1. User settings

noise_level = 0.7;               % Fraction of valid entries corrupted, e.g. 0.80 = 80%.
number_of_sine_modes = 500;      % First N sin(x)sin(y) modes in library.
number_of_laplacian_modes = 500; % First N graph-Laplacian modes in library.

% 2. Demo settings

cfg = struct();
cfg.random_seed = 10;
cfg.snapshot_index = 1;
cfg.corruption_amplitude_multiplier = 10;

% RPCA parameters
cfg.rpca = struct();
cfg.rpca.lambda = 1;
cfg.rpca.convergence_tolerance = 1e-6;
cfg.rpca.maximum_iterations = 1000;

% LLA-RPCA parameters
cfg.llrpca = struct();
cfg.llrpca.active_rank = 5;
cfg.llrpca.convergence_tolerance = 1e-6;
cfg.llrpca.field_change_tolerance = 1e-6;
cfg.llrpca.maximum_iterations = 1000;
cfg.llrpca.nu_initial = 0.01;
cfg.llrpca.nu_growth_factor = 1.05;
cfg.llrpca.nu_maximum = 1e6;
cfg.llrpca.print = true;
cfg.llrpca.n_sine_modes = number_of_sine_modes;
cfg.llrpca.n_laplacian_modes = number_of_laplacian_modes;

% Computational domain
cfg.x_phys_lim = [-1.3864142538975501, 3.9365256124721606];
cfg.y_phys_lim = [-1.1959798994974875, 1.1959798994974875];

% Visualization settings
cfg.visualization_mask_expansion_pixels = 0.75;
cfg.visualization_mask_supersampling = 8;
cfg.visualization_render_scale = 16;
cfg.disable_geometry_overlay = false;
cfg.gif_noise_clip_multiplier = 2.0;

% File directories
root_dir = fileparts(mfilename('fullpath'));
cfg.data_file = resolve_demo_file(root_dir, 'NACA0012_a40_VelocityY.mat');
cfg.geometry_file = resolve_demo_file(root_dir, 'geometry.inp');

% 3. Load and prepare data

loaded = load(cfg.data_file);
if isfield(loaded, 'data')
    data = loaded.data;
else
    variable_names = fieldnames(loaded);
    data = loaded.(variable_names{1});
end

finite_values = data(isfinite(data));
scale_factor = max(abs(double(finite_values(:))));
if isempty(scale_factor) || ~isfinite(scale_factor) || scale_factor <= 0
    scale_factor = 1;
end
data = data ./ scale_factor;

data = data(1:size(data, 1), :, :);

[nt, ny, nx] = size(data);
cfg.snapshot_index = min(max(round(cfg.snapshot_index), 1), nt);

geometry = build_geometry_and_masks(data, cfg.geometry_file, nx, ny, cfg);
valid_fluid_mask = geometry.valid_fluid_mask;

X_clean = flow_array_to_matrix(data);
X_clean(~valid_fluid_mask(:), :) = 0;


% 4. Add artificial sparse corruption

rng(cfg.random_seed);
X_noisy = add_signed_sparse_spikes( ...
    X_clean, valid_fluid_mask, noise_level, cfg.corruption_amplitude_multiplier);


% 5. Run standard RPCA and LLA-RPCA

fprintf('Running RPCA/LLA-RPCA example...\n');
fprintf('  noise level       : %.2f\n', noise_level);
fprintf('  sine modes        : %d\n', cfg.llrpca.n_sine_modes);
fprintf('  Laplacian modes   : %d\n', cfg.llrpca.n_laplacian_modes);
fprintf('  frames            : %d\n', nt);
fprintf('  grid              : %d x %d\n', ny, nx);

results = llrpca_solver(X_noisy, valid_fluid_mask, nx, ny, cfg);

rpca_error = relative_error(X_clean, results.rpca.L, valid_fluid_mask);
llrpca_error = relative_error(X_clean, results.llrpca.L, valid_fluid_mask);

fprintf('\nFinished.\n');
fprintf('  RPCA relative error     : %.4e\n', rpca_error);
fprintf('  LLA-RPCA relative error : %.4e\n', llrpca_error);


% 6. Plot one snapshot comparison

plot_result( ...
    X_clean, X_noisy, results.rpca.L, results.rpca.S, ...
    results.llrpca.L, results.llrpca.S, ...
    geometry, ny, nx, cfg.snapshot_index, cfg);


% Helper functions

function filepath = resolve_demo_file(root_dir, filename)
    candidates = { ...
        fullfile(root_dir, filename), ...
        fullfile(root_dir, 'data', filename), ...
        fullfile(pwd, filename), ...
        fullfile(pwd, 'data', filename)};

    filepath = '';
    for k = 1:numel(candidates)
        if isfile(candidates{k})
            filepath = candidates{k};
            return;
        end
    end

    if endsWith(filename, '.inp')
        return;
    end
end

function geometry = build_geometry_and_masks(data, geometry_file, nx, ny, cfg)
    x_phys_lim = cfg.x_phys_lim;
    y_phys_lim = cfg.y_phys_lim;

    geometry_raw = readmatrix(geometry_file, 'NumHeaderLines', 1, 'FileType', 'text');

    % Match main_velocity.m: retain finite geometry points and ignore
    % terminal/sentinel rows carrying a negative geometry ID.
    geometry_raw = geometry_raw( ...
        isfinite(geometry_raw(:, 1)) & isfinite(geometry_raw(:, 2)), :);

    if size(geometry_raw, 2) >= 3
        geometry_raw = geometry_raw(geometry_raw(:, 3) >= 0 | isnan(geometry_raw(:, 3)), :);
    end

    gx = geometry_raw(:, 1);
    gy = geometry_raw(:, 2);

    if size(geometry_raw, 2) >= 3
        geometry_ids = geometry_raw(:, 3);
        geometry_ids(~isfinite(geometry_ids)) = 0;
    else
        geometry_ids = zeros(size(gx));
    end

    unique_ids = unique(geometry_ids(:));
    unique_ids = unique_ids(~isnan(unique_ids));

    if any(unique_ids > 0)
        solid_ids = unique_ids(unique_ids > 0);
    else
        solid_ids = 0;
    end

    if any(geometry_ids == 2)
        wing_id = 2;
    else
        wing_id = solid_ids(1);
    end

    geometry_polygons = struct( ...
        'id', {}, 'xpix', {}, 'ypix', {}, 'xphys', {}, 'yphys', {});

    for k = 1:numel(solid_ids)
        current_id = solid_ids(k);
        selection = geometry_ids == current_id;

        x = gx(selection);
        y = gy(selection);

        xpix = phys_to_pix_x(x, x_phys_lim, nx);
        ypix = phys_to_pix_y(y, y_phys_lim, ny);

        geometry_polygons(k).id = current_id;
        geometry_polygons(k).xpix = xpix;
        geometry_polygons(k).ypix = ypix;
        geometry_polygons(k).xphys = x;
        geometry_polygons(k).yphys = y;
    end

    solid_mask = false(ny, nx);
    for k = 1:numel(geometry_polygons)
        mask_this_object = poly2mask( ...
            geometry_polygons(k).xpix, ...
            geometry_polygons(k).ypix, ny, nx);
        solid_mask = solid_mask | flipud(mask_this_object);
    end

    wing_mask = false(ny, nx);
    for k = 1:numel(geometry_polygons)
        if geometry_polygons(k).id == wing_id
            mask_this_object = poly2mask( ...
                geometry_polygons(k).xpix, ...
                geometry_polygons(k).ypix, ny, nx);
            wing_mask = wing_mask | flipud(mask_this_object);
        end
    end

    visualization_solid_mask = expand_binary_mask_2d( ...
        solid_mask, cfg.visualization_mask_expansion_pixels);
    visualization_wing_mask = expand_binary_mask_2d( ...
        wing_mask, cfg.visualization_mask_expansion_pixels);

    valid_fluid_mask = ~solid_mask;

    geometry = struct();
    geometry.geometry_polygons = geometry_polygons;
    geometry.solid_mask = solid_mask;
    geometry.valid_fluid_mask = valid_fluid_mask;
    geometry.wing_mask = wing_mask;
    geometry.visualization_solid_mask = visualization_solid_mask;
    geometry.visualization_wing_mask = visualization_wing_mask;
    geometry.visualization_mask_expansion_pixels = ...
        cfg.visualization_mask_expansion_pixels;
    geometry.candidate_indices = find(valid_fluid_mask);
    geometry.number_of_candidates = nnz(valid_fluid_mask);
    geometry.x_phys_lim = x_phys_lim;
    geometry.y_phys_lim = y_phys_lim;
end

function expanded_mask = expand_binary_mask_2d(mask_in, expansion_pixels)
    mask_in = logical(mask_in);

    if isempty(mask_in) || expansion_pixels <= 0
        expanded_mask = mask_in;
        return;
    end

    if exist('bwdist', 'file') == 2
        expanded_mask = bwdist(mask_in) <= expansion_pixels;
        return;
    end

    kernel_radius = ceil(expansion_pixels);
    [kernel_x, kernel_y] = meshgrid( ...
        -kernel_radius:kernel_radius, ...
        -kernel_radius:kernel_radius);
    kernel = hypot(kernel_x, kernel_y) <= expansion_pixels;
    expanded_mask = conv2(double(mask_in), double(kernel), 'same') > 0;
end

function xpix = phys_to_pix_x(x, x_phys_lim, nx)
    xmin = x_phys_lim(1);
    xmax = x_phys_lim(2);
    xpix = (x - xmin) ./ (xmax - xmin) * (nx - 1) + 1;
end

function ypix = phys_to_pix_y(y, y_phys_lim, ny)
    ymin = y_phys_lim(1);
    ymax = y_phys_lim(2);
    ypix = (ymax - y) ./ (ymax - ymin) * (ny - 1) + 1;
end

function X = flow_array_to_matrix(data)
    [nt, ny, nx] = size(data);
    X = reshape(permute(data, [2, 3, 1]), ny * nx, nt);
    X(~isfinite(X)) = 0;
end

function X_noisy = add_signed_sparse_spikes(X_clean, valid_fluid_mask, noise_level, amplitude_multiplier)
    X_noisy = X_clean;
    if noise_level <= 0
        return;
    end

    valid_linear_mask = valid_fluid_mask(:);
    valid_entries = find(repmat(valid_linear_mask, 1, size(X_clean, 2)));
    n_corrupted = round(noise_level * numel(valid_entries));
    if n_corrupted <= 0
        return;
    end

    valid_values = X_clean(valid_linear_mask, :);
    corruption_amplitude = amplitude_multiplier * std(valid_values(:));

    corrupted_entries = valid_entries(randperm(numel(valid_entries), n_corrupted));
    n_positive = floor(n_corrupted / 2);

    X_noisy(corrupted_entries(1:n_positive)) = corruption_amplitude;
    X_noisy(corrupted_entries(n_positive + 1:end)) = -corruption_amplitude;
end

function err = relative_error(X_reference, X_approx, valid_fluid_mask)
    valid_linear_mask = valid_fluid_mask(:);
    err = norm(X_reference(valid_linear_mask, :) - X_approx(valid_linear_mask, :), 'fro') / ...
        max(norm(X_reference(valid_linear_mask, :), 'fro'), eps);
end

function plot_result(X_clean, X_noisy, L_rpca, S_rpca, L_llrpca, S_llrpca, geometry, ny, nx, snapshot_index, cfg)
    clean_frame = matrix_column_to_frame(X_clean, geometry, ny, nx, snapshot_index);
    noisy_frame = matrix_column_to_frame(X_noisy, geometry, ny, nx, snapshot_index);
    L_rpca_frame = matrix_column_to_frame(L_rpca, geometry, ny, nx, snapshot_index);
    S_rpca_frame = matrix_column_to_frame(S_rpca, geometry, ny, nx, snapshot_index);
    L_llrpca_frame = matrix_column_to_frame(L_llrpca, geometry, ny, nx, snapshot_index);
    S_llrpca_frame = matrix_column_to_frame(S_llrpca, geometry, ny, nx, snapshot_index);

    shared_velocity_display_limit = compute_shared_snapshot_abs_display_limit( ...
        {clean_frame, L_rpca_frame, L_llrpca_frame}, ...
        geometry.solid_mask);

    soft_limits = [-shared_velocity_display_limit, shared_velocity_display_limit];
    hard_limits = cfg.gif_noise_clip_multiplier * soft_limits;

    figure('Name', 'Toy LLA-RPCA example', 'Color', 'w', 'Visible', 'on');
    tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    plot_panel(clean_frame, 'Reference', soft_limits, hard_limits, geometry, cfg);
    plot_panel(L_rpca_frame, 'RPCA L', soft_limits, hard_limits, geometry, cfg);
    plot_panel(S_rpca_frame, 'RPCA S', soft_limits, hard_limits, geometry, cfg);

    plot_panel(noisy_frame, 'Reference + noise', soft_limits, hard_limits, geometry, cfg);
    plot_panel(L_llrpca_frame, 'LLA-RPCA L', soft_limits, hard_limits, geometry, cfg);
    plot_panel(S_llrpca_frame, 'LLA-RPCA S', soft_limits, hard_limits, geometry, cfg);

    sgtitle('RPCA vs LLA-RPCA');
    drawnow;
end

function frame = matrix_column_to_frame(X, geometry, ny, nx, snapshot_index)
    frame = reshape(X(:, snapshot_index), ny, nx);
    frame(geometry.solid_mask) = NaN;
end

function shared_limit = compute_shared_snapshot_abs_display_limit(frame_collection, mask_2d)
    shared_limit = 0;

    for i_frame = 1:numel(frame_collection)
        frame = double(frame_collection{i_frame});
        valid_mask = isfinite(frame);

        if nargin >= 2 && ~isempty(mask_2d)
            valid_mask = valid_mask & ~logical(mask_2d);
        end

        values = frame(valid_mask);
        if ~isempty(values)
            shared_limit = max(shared_limit, max(abs(values(:))));
        end
    end

    if ~isfinite(shared_limit) || shared_limit <= 0
        shared_limit = 1;
    end
end

function plot_panel(frame, title_text, soft_limits, hard_limits, geometry, cfg)
    nexttile;

    rgb_native = map_frame_to_buffered_rgb( ...
        frame, geometry.solid_mask, soft_limits, hard_limits);

    render_scale = 1;
    if isfield(cfg, 'visualization_render_scale') && ...
            ~isempty(cfg.visualization_render_scale) && ...
            isfinite(cfg.visualization_render_scale)
        render_scale = max(1, round(double(cfg.visualization_render_scale)));
    end

    if render_scale > 1
        if exist('imresize', 'file') == 2
            rgb_image = imresize(rgb_native, render_scale, 'nearest');
        else
            rgb_image = repelem(rgb_native, render_scale, render_scale, 1);
        end
    else
        rgb_image = rgb_native;
    end

    rgb_image = apply_smooth_geometry_overlay( ...
        rgb_image, geometry, cfg, [1, 1, 1]);

    imagesc(rgb_image);
    set(gca, 'YDir', 'reverse');
    axis equal tight off;
    title(title_text, 'Interpreter', 'none');

    ax = gca;

    colorbar_cmap = demo_red_white_blue_with_outlier_caps(256, 18);
    colormap(ax, colorbar_cmap);
    clim(ax, soft_limits);

    cb = colorbar;
    cb.Label.String = 'v / max |v|';
end

function rgb_image = map_frame_to_buffered_rgb(frame_in, solid_mask, soft_limits, hard_limits)
    frame_plot = flipud(double(frame_in));
    solid_mask_plot = flipud(logical(solid_mask));

    fluid_cmap = demo_red_white_blue_core(256);
    green_color = [0, 1, 0];
    black_color = [0, 0, 0];

    [nrows, ncols] = size(frame_plot);
    rgb_image = ones(nrows, ncols, 3);

    nan_mask = isnan(frame_plot) | solid_mask_plot;
    valid_mask = ~nan_mask;

    high_clip_mask = frame_plot > hard_limits(2) & valid_mask;
    low_clip_mask = frame_plot < hard_limits(1) & valid_mask;
    regular_mask = valid_mask & ~high_clip_mask & ~low_clip_mask;

    clipped_values = frame_plot;
    clipped_values(clipped_values < soft_limits(1)) = soft_limits(1);
    clipped_values(clipped_values > soft_limits(2)) = soft_limits(2);

    value_range = soft_limits(2) - soft_limits(1);
    if value_range <= 0
        value_range = 1;
    end

    normalized = (clipped_values - soft_limits(1)) / value_range;
    normalized = max(min(normalized, 1), 0);

    idx = 1 + round(normalized * (size(fluid_cmap, 1) - 1));
    idx = max(min(idx, size(fluid_cmap, 1)), 1);

    for c = 1:3
        channel = ones(nrows, ncols);
        cmap_c = fluid_cmap(:, c);
        channel(regular_mask) = cmap_c(idx(regular_mask));
        channel(high_clip_mask) = green_color(c);
        channel(low_clip_mask) = black_color(c);
        channel(nan_mask) = 1;
        rgb_image(:, :, c) = channel;
    end
end

function cmap = demo_red_white_blue_core(n)
    if nargin < 1 || isempty(n)
        n = 256;
    end

    n = max(round(n), 8);
    n_neg = floor(n / 2);
    n_pos = n - n_neg;

    blue = [0, 0, 1];
    white = [1, 1, 1];
    red = [1, 0, 0];

    neg_branch = [linspace(blue(1), white(1), n_neg).', ...
                  linspace(blue(2), white(2), n_neg).', ...
                  linspace(blue(3), white(3), n_neg).'];

    pos_branch = [linspace(white(1), red(1), n_pos).', ...
                  linspace(white(2), red(2), n_pos).', ...
                  linspace(white(3), red(3), n_pos).'];

    cmap = [neg_branch; pos_branch];
end

function cmap = demo_red_white_blue_with_outlier_caps(n_total, n_clip)
    if nargin < 1 || isempty(n_total)
        n_total = 256;
    end
    if nargin < 2 || isempty(n_clip)
        n_clip = 18;
    end

    n_total = max(round(n_total), 32);
    n_clip = max(round(n_clip), 2);
    n_core = max(n_total - 2 * n_clip, 8);

    core = demo_red_white_blue_core(n_core);

    black_color = [0, 0, 0];
    green_color = [0, 1, 0];

    black_clip = repmat(black_color, n_clip, 1);
    yellow_clip = repmat(green_color, n_clip, 1);

    cmap = [black_clip; core; yellow_clip];

    if size(cmap, 1) ~= n_total
        x_old = linspace(0, 1, size(cmap, 1));
        x_new = linspace(0, 1, n_total);
        cmap = interp1(x_old, cmap, x_new, 'linear');
    end
end

function rgb_out = apply_smooth_geometry_overlay(rgb_in, geometry, cfg, face_color)
    if nargin < 4 || isempty(face_color)
        face_color = [1, 1, 1];
    end

    rgb_out = rgb_in;
    if isempty(rgb_in) || ndims(rgb_in) ~= 3 || size(rgb_in, 3) ~= 3
        return;
    end

    if isfield(cfg, 'disable_geometry_overlay') && logical(cfg.disable_geometry_overlay)
        return;
    end

    target_ny = size(rgb_in, 1);
    target_nx = size(rgb_in, 2);
    alpha = build_smooth_geometry_alpha(geometry, cfg, target_ny, target_nx);

    if isempty(alpha) || ~any(alpha(:) > 0)
        return;
    end

    alpha = max(min(double(alpha), 1), 0);
    face_color = max(min(double(face_color(:).'), 1), 0);

    for c = 1:3
        rgb_out(:, :, c) = ...
            (1 - alpha) .* rgb_out(:, :, c) + alpha .* face_color(c);
    end
end

function alpha = build_smooth_geometry_alpha(geometry, cfg, target_ny, target_nx)
    computational_mask = geometry.solid_mask;
    [source_ny, source_nx] = size(computational_mask);

    requested_supersampling = max(1, round(double(cfg.visualization_mask_supersampling)));
    expansion_source_pixels = max(0, double(cfg.visualization_mask_expansion_pixels));

    scale_x = max(target_nx - 1, 1) / max(source_nx - 1, 1);
    scale_y = max(target_ny - 1, 1) / max(source_ny - 1, 1);
    expansion_target_pixels = expansion_source_pixels * sqrt(scale_x * scale_y);

    is_native_size = target_ny == source_ny && target_nx == source_nx;
    if is_native_size
        supersampling = requested_supersampling;
    elseif target_ny * target_nx <= 1e6
        supersampling = min(requested_supersampling, 2);
    else
        supersampling = 1;
    end

    high_ny = target_ny * supersampling;
    high_nx = target_nx * supersampling;
    mask_high = false(high_ny, high_nx);

    for k = 1:numel(geometry.geometry_polygons)
        xpix = double(geometry.geometry_polygons(k).xpix(:));
        ypix = double(geometry.geometry_polygons(k).ypix(:));

        valid_points = isfinite(xpix) & isfinite(ypix);
        xpix = xpix(valid_points);
        ypix = ypix(valid_points);

        if numel(xpix) < 3
            continue;
        end

        n_resample = max(800, 8 * numel(xpix));
        [xpix, ypix] = resample_closed_polygon_by_arclength(xpix, ypix, n_resample);

        x_target = 1 + (xpix - 1) * scale_x;
        y_target = 1 + (ypix - 1) * scale_y;

        x_high = (x_target - 0.5) * supersampling + 0.5;
        y_high = (y_target - 0.5) * supersampling + 0.5;

        if exist('poly2mask', 'file') == 2
            polygon_mask = poly2mask(x_high, y_high, high_ny, high_nx);
        else
            [x_grid, y_grid] = meshgrid(1:high_nx, 1:high_ny);
            polygon_mask = inpolygon(x_grid, y_grid, x_high, y_high);
        end

        mask_high = mask_high | polygon_mask;
    end

    expansion_high = expansion_target_pixels * supersampling;
    if expansion_high > 0 && any(mask_high(:))
        if exist('bwdist', 'file') == 2
            mask_high = bwdist(mask_high) <= expansion_high;
        else
            kernel_radius = ceil(expansion_high);
            [kernel_x, kernel_y] = meshgrid( ...
                -kernel_radius:kernel_radius, ...
                -kernel_radius:kernel_radius);
            kernel = hypot(kernel_x, kernel_y) <= expansion_high;
            mask_high = conv2(double(mask_high), double(kernel), 'same') > 0;
        end
    end

    if supersampling > 1
        mask_blocks = reshape( ...
            double(mask_high), supersampling, target_ny, ...
            supersampling, target_nx);
        alpha = squeeze(mean(mean(mask_blocks, 1), 3));
        alpha = reshape(alpha, target_ny, target_nx);
    else
        alpha = double(mask_high);
        if exist('imgaussfilt', 'file') == 2
            alpha = imgaussfilt(alpha, 0.55, 'FilterSize', 3, 'Padding', 'replicate');
        else
            antialias_kernel = [1, 2, 1; 2, 4, 2; 1, 2, 1] / 16;
            alpha = conv2(alpha, antialias_kernel, 'same');
        end
    end

    alpha = max(min(alpha, 1), 0);
end

function [x_resampled, y_resampled] = resample_closed_polygon_by_arclength(x_in, y_in, n_points)
    if nargin < 3 || isempty(n_points)
        n_points = 300;
    end

    x_in = x_in(:);
    y_in = y_in(:);

    if numel(x_in) < 3
        x_resampled = x_in;
        y_resampled = y_in;
        return;
    end

    if hypot(x_in(1) - x_in(end), y_in(1) - y_in(end)) > 1e-10
        x_in(end + 1) = x_in(1);
        y_in(end + 1) = y_in(1);
    end

    ds = hypot(diff(x_in), diff(y_in));
    keep_segment = ds > 1e-12;

    x_clean = x_in([true; keep_segment]);
    y_clean = y_in([true; keep_segment]);

    if numel(x_clean) < 3
        x_resampled = x_in;
        y_resampled = y_in;
        return;
    end

    s_clean = [0; cumsum(hypot(diff(x_clean), diff(y_clean)))];
    total_length = s_clean(end);

    if ~isfinite(total_length) || total_length <= 0
        x_resampled = x_clean;
        y_resampled = y_clean;
        return;
    end

    s_query = linspace(0, total_length, max(20, round(n_points))).';
    x_resampled = interp1(s_clean, x_clean, s_query, 'pchip');
    y_resampled = interp1(s_clean, y_clean, s_query, 'pchip');
end
