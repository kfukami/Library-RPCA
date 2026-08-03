% lla_rpca_demo_solver.m
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


function [L, S, info] = lla_rpca_demo_solver( ...
        X, active_rank, number_of_sine_modes, ...
        number_of_laplacian_modes, valid_fluid_mask, options)
% Library-constrained RPCA demonstration solver.
%
%   L = Psi * B * A
%
% The solver minimizes ||S||_1 subject to
%
%   X = Psi * B * A + S.

    convergence_tolerance = ...
        options.convergence_tolerance;

    field_change_tolerance = ...
        options.field_change_tolerance;

    maximum_iterations = ...
        options.maximum_iterations;

    nu = options.nu_initial;
    nu_growth_factor = options.nu_growth_factor;
    nu_maximum = options.nu_maximum;

    X_norm = max(norm(X, 'fro'), eps);

    %% Candidate library

    Psi = build_candidate_library( ...
        valid_fluid_mask, ...
        number_of_sine_modes, ...
        number_of_laplacian_modes);

    library_rank = size(Psi, 2);
    number_of_snapshots = size(X, 2);

    %% Initialization

    B = zeros(library_rank, active_rank);
    B(1:active_rank, :) = eye(active_rank);

    Phi = Psi * B;

    A = zeros(active_rank, number_of_snapshots);
    L = zeros(size(X));
    S = zeros(size(X));
    Y = zeros(size(X));

    stop_reason = 'maximum_iterations';

    %% Alternating updates

    for iteration = 1:maximum_iterations

        L_old = L;
        S_old = S;

        shifted_data = X - S + Y / nu;

        % Update A
        A = Phi' * shifted_data;

        % Update B
        coefficient_matrix = A * A';
        right_hand_side = Psi' * shifted_data * A';

        if rcond(coefficient_matrix) > 1e-12
            B = right_hand_side / coefficient_matrix;
        else
            B = right_hand_side * pinv(coefficient_matrix);
        end

        % QR reparameterization
        [Phi, R] = qr(Psi * B, 0);

        B = B / R;
        A = R * A;

        % Update L
        L = Phi * A;

        % Update S
        S = soft_threshold( ...
            X - L + Y / nu, ...
            1 / nu);

        % Update Y
        primal_residual = X - L - S;
        Y = Y + nu * primal_residual;

        relative_primal_residual = ...
            norm(primal_residual, 'fro') / X_norm;

        relative_change_L = ...
            norm(L - L_old, 'fro') ...
            / max(norm(L_old, 'fro'), eps);

        relative_change_S = ...
            norm(S - S_old, 'fro') ...
            / max(norm(S_old, 'fro'), eps);

        field_change = max( ...
            relative_change_L, relative_change_S);

        if relative_primal_residual < convergence_tolerance ...
                && field_change < field_change_tolerance

            stop_reason = 'converged';
            break;
        end

        nu = min( ...
            nu_growth_factor * nu, ...
            nu_maximum);
    end

    info.iterations = iteration;
    info.stop_reason = stop_reason;

    info.relative_primal_residual = ...
        relative_primal_residual;

    info.library_rank = library_rank;
end

function Psi = build_candidate_library( ...
        valid_fluid_mask, ...
        number_of_sine_modes, ...
        number_of_laplacian_modes)

    [edge_start, edge_end] = ...
        build_fluid_neighbor_pairs(valid_fluid_mask);

    sine_candidates = build_sine_candidates( ...
        valid_fluid_mask, number_of_sine_modes);

    laplacian_candidates = build_laplacian_candidates( ...
        valid_fluid_mask, number_of_laplacian_modes, ...
        edge_start, edge_end);

    sine_candidates = sine_candidates ...
        - mean(sine_candidates, 1);

    if size(laplacian_candidates, 2) > 1

        laplacian_candidates(:, 2:end) = ...
            laplacian_candidates(:, 2:end) ...
            - mean(laplacian_candidates(:, 2:end), 1);
    end

    candidates = [ ...
        sine_candidates, laplacian_candidates];

    roughness = graph_dirichlet_roughness( ...
        candidates, edge_start, edge_end);

    [~, order] = sort(roughness);
    candidates = candidates(:, order);

    [Psi, R] = qr(candidates, 0);

    diagonal_R = abs(diag(R));

    numerical_tolerance = ...
        1e-8 * max(diagonal_R);

    library_rank = sum( ...
        diagonal_R > numerical_tolerance);

    Psi = Psi(:, 1:library_rank);
end

function candidates = build_sine_candidates( ...
        valid_fluid_mask, number_of_modes)

    [ny, nx] = size(valid_fluid_mask);

    wave_pairs = ...
        generate_diagonal_wave_pairs(number_of_modes);

    x_hat = ((1:nx) - 0.5) / nx;
    y_hat = ((1:ny) - 0.5) / ny;

    candidates = zeros( ...
        nnz(valid_fluid_mask), number_of_modes);

    for i_mode = 1:number_of_modes

        kx = wave_pairs(i_mode, 1);
        ky = wave_pairs(i_mode, 2);

        x_mode = sin(pi * kx * x_hat);
        y_mode = sin(pi * ky * y_hat);

        mode_grid = y_mode(:) * x_mode(:)';

        candidates(:, i_mode) = ...
            mode_grid(valid_fluid_mask);
    end
end

function candidates = build_laplacian_candidates( ...
        valid_fluid_mask, number_of_modes, ...
        edge_start, edge_end)

    number_of_fluid_points = nnz(valid_fluid_mask);

    adjacency = sparse( ...
        [edge_start; edge_end], ...
        [edge_end; edge_start], ...
        1, ...
        number_of_fluid_points, ...
        number_of_fluid_points);

    degree = full(sum(adjacency, 2));

    laplacian = spdiags( ...
        degree, 0, ...
        number_of_fluid_points, ...
        number_of_fluid_points) ...
        - adjacency;

    laplacian = 0.5 * (laplacian + laplacian');

    eigs_options.tol = 1e-10;
    eigs_options.maxit = 5000;

    [candidates, eigenvalue_matrix] = eigs( ...
        laplacian, number_of_modes, ...
        'smallestreal', eigs_options);

    eigenvalues = real(diag(eigenvalue_matrix));

    [~, order] = sort(eigenvalues, 'ascend');

    candidates = real(candidates(:, order));

    candidates = candidates ./ vecnorm(candidates, 2, 1);

    [candidates, ~] = qr(candidates, 0);
end

function wave_pairs = ...
        generate_diagonal_wave_pairs(number_of_pairs)

    wave_pairs = zeros(number_of_pairs, 2);

    pair_count = 0;
    diagonal_sum = 2;

    while pair_count < number_of_pairs

        for kx = 1:(diagonal_sum - 1)

            ky = diagonal_sum - kx;

            pair_count = pair_count + 1;

            wave_pairs(pair_count, :) = [kx, ky];

            if pair_count == number_of_pairs
                return;
            end
        end

        diagonal_sum = diagonal_sum + 1;
    end
end

function [edge_start, edge_end] = ...
        build_fluid_neighbor_pairs(valid_fluid_mask)

    [ny, nx] = size(valid_fluid_mask);

    fluid_index = zeros(ny, nx);

    fluid_index(valid_fluid_mask) = ...
        1:nnz(valid_fluid_mask);

    horizontal_edges = ...
        valid_fluid_mask(:, 1:end-1) ...
        & valid_fluid_mask(:, 2:end);

    vertical_edges = ...
        valid_fluid_mask(1:end-1, :) ...
        & valid_fluid_mask(2:end, :);

    horizontal_left = fluid_index(:, 1:end-1);
    horizontal_right = fluid_index(:, 2:end);

    vertical_lower = fluid_index(1:end-1, :);
    vertical_upper = fluid_index(2:end, :);

    edge_start = [ ...
        horizontal_left(horizontal_edges); ...
        vertical_lower(vertical_edges)];

    edge_end = [ ...
        horizontal_right(horizontal_edges); ...
        vertical_upper(vertical_edges)];
end

function roughness = graph_dirichlet_roughness( ...
        candidates, edge_start, edge_end)

    number_of_modes = size(candidates, 2);
    roughness = zeros(1, number_of_modes);

    for i_mode = 1:number_of_modes

        mode = candidates(:, i_mode);

        differences = ...
            mode(edge_start) - mode(edge_end);

        roughness(i_mode) = ...
            sum(differences.^2) / sum(mode.^2);
    end
end

function Z = soft_threshold(X, threshold)

    Z = sign(X) .* max(abs(X) - threshold, 0);
end
