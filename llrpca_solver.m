% llrpca_solver.m
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


function results = llrpca_solver(X_noisy_full, valid_fluid_mask, nx, ny, cfg)

    valid_linear_mask = valid_fluid_mask(:);
    X_noisy = X_noisy_full(valid_linear_mask, :);

    [L_rpca, S_rpca, iter_rpca, info_rpca] = rpca_ialm( ...
        X_noisy, cfg.rpca.lambda, ...
        cfg.rpca.convergence_tolerance, cfg.rpca.maximum_iterations);

    [L_llrpca, S_llrpca, iter_llrpca, info_llrpca] = llrpca_alm( ...
        X_noisy, valid_fluid_mask, nx, ny, cfg.llrpca);

    results = struct();
    results.rpca.L = expand_to_full_grid(L_rpca, valid_fluid_mask);
    results.rpca.S = expand_to_full_grid(S_rpca, valid_fluid_mask);
    results.rpca.iterations = iter_rpca;
    results.rpca.info = info_rpca;

    results.llrpca.L = expand_to_full_grid(L_llrpca, valid_fluid_mask);
    results.llrpca.S = expand_to_full_grid(S_llrpca, valid_fluid_mask);
    results.llrpca.iterations = iter_llrpca;
    results.llrpca.info = info_llrpca;
end

function [L, S, iter, info] = llrpca_alm(X, valid_fluid_mask, nx, ny, llcfg)
    [m, n] = size(X);

    Psi_full = build_secondary_library( ...
        nx, ny, valid_fluid_mask, llcfg.n_sine_modes, llcfg.n_laplacian_modes);
    Psi = Psi_full(valid_fluid_mask(:), :);

    projected_singular_values = svd(Psi' * X, 'econ');
    active_rank = choose_active_rank(projected_singular_values, llcfg.active_rank);
    active_rank = min(active_rank, size(Psi, 2));
    active_rank = max(active_rank, 1);

    B = zeros(size(Psi, 2), active_rank);
    B(1:active_rank, 1:active_rank) = eye(active_rank);

    Phi = Psi * B;
    A = zeros(active_rank, n);
    S = zeros(m, n);
    Y = zeros(m, n);
    L = Phi * A;

    PsiTPsi = Psi' * Psi;
    orthogonality_tolerance = 1e-8;
    psi_orthogonality_error = norm( ...
        PsiTPsi - speye(size(PsiTPsi, 1)), 'fro') / ...
        max(sqrt(size(PsiTPsi, 1)), eps);

    if isfield(llcfg, 'field_change_tolerance')
        field_change_tolerance = llcfg.field_change_tolerance;
    else
        field_change_tolerance = 1e-5;
    end

    nu = llcfg.nu_initial;
    converged = false;
    stagnated = false;
    stop_reason = 'maximum iterations';

    for iter = 1:llcfg.maximum_iterations
        A_old = A;
        B_old = B;
        S_old = S;
        L_old = L;

        shifted_data = X - S + (1 / nu) * Y;

        PhiTPhi = Phi' * Phi;
        phi_orthogonality_error = norm( ...
            PhiTPhi - speye(size(PhiTPhi, 1)), 'fro') / ...
            max(sqrt(size(PhiTPhi, 1)), eps);

        if phi_orthogonality_error <= orthogonality_tolerance
            A = Phi' * shifted_data;
        else
            A = Phi \ shifted_data;
        end

        AAT = A * A';
        RHS_B = Psi' * shifted_data * A';

        if psi_orthogonality_error > orthogonality_tolerance
            error('Psi is not sufficiently orthonormal for the B-update.');
        end

        B = solve_right_unregularized(RHS_B, AAT);

        Phi = Psi * B;
        [Phi_orth, R] = qr(Phi, 0);

        if rcond(full(R)) <= eps(max(size(R)))
            error('The learned basis lost column rank during QR reparameterization.');
        end

        B = B / R;
        A = R * A;
        Phi = Phi_orth;

        L = Phi * A;
        S = soft_threshold(X - L + (1 / nu) * Y, 1 / nu);

        primal_residual = X - L - S;
        Y = Y + nu * primal_residual;

        relative_primal_residual = norm(primal_residual, 'fro') / max(norm(X, 'fro'), eps);
        relative_change_A = norm(A - A_old, 'fro') / max(norm(A_old, 'fro'), eps);
        relative_change_B = norm(B - B_old, 'fro') / max(norm(B_old, 'fro'), eps);
        relative_change_S = norm(S - S_old, 'fro') / max(norm(S_old, 'fro'), eps);
        relative_change_L = norm(L - L_old, 'fro') / max(norm(L_old, 'fro'), eps);

        coefficient_change = max(relative_change_A, relative_change_B);
        field_change = max(relative_change_L, relative_change_S);
        combined_change = max(coefficient_change, field_change);

        if llcfg.print && (iter == 1 || mod(iter, 25) == 0)
            fprintf('  LLA-RPCA iter %4d | rel primal %.3e | rel field %.3e | rel coeff %.3e | nu %.3e\n', ...
                iter, relative_primal_residual, field_change, coefficient_change, nu);
        end

        if relative_primal_residual < llcfg.convergence_tolerance && ...
                combined_change < llcfg.convergence_tolerance
            converged = true;
            stop_reason = 'converged_strict';
            break;
        end

        if relative_primal_residual < llcfg.convergence_tolerance && ...
                field_change < field_change_tolerance
            converged = true;
            stop_reason = 'converged_field';
            break;
        end

        if field_change < llcfg.convergence_tolerance
            stagnated = true;
            stop_reason = 'field_stagnated';
            break;
        end

        nu = min(llcfg.nu_growth_factor * nu, llcfg.nu_maximum);
    end

    info = struct();
    info.Psi = Psi;
    info.Phi = Phi;
    info.A = A;
    info.B = B;
    info.active_rank = active_rank;
    info.projected_singular_values = projected_singular_values;
    info.psi_orthogonality_error = psi_orthogonality_error;
    info.converged = converged;
    info.stagnated = stagnated;
    info.stop_reason = stop_reason;
    info.nu_final = nu;
    info.final_relative_primal_residual = relative_primal_residual;
    info.final_field_change = field_change;
    info.final_coefficient_change = coefficient_change;
    info.final_combined_change = combined_change;
    info.field_change_tolerance = field_change_tolerance;
end

function Psi = build_secondary_library(nx, ny, valid_fluid_mask, n_sine_modes, n_laplacian_modes)
    Psi_sine = build_sinx_siny_modes(nx, ny, valid_fluid_mask, n_sine_modes);
    Psi_laplacian = build_graph_laplacian_modes(nx, ny, valid_fluid_mask, n_laplacian_modes);

    Psi_candidate = [Psi_sine, Psi_laplacian];

    remove_mean_sine = true(1, size(Psi_sine, 2));
    remove_mean_laplacian = true(1, size(Psi_laplacian, 2));
    if ~isempty(remove_mean_laplacian)
        remove_mean_laplacian(1) = false;
    end
    remove_mean_flags = [remove_mean_sine, remove_mean_laplacian];

    % Order by roughness
    roughness_scores = compute_candidate_graph_roughness_scores( ...
        Psi_candidate, valid_fluid_mask, remove_mean_flags, nx, ny);
    original_positions = (1:size(Psi_candidate, 2)).';
    roughness_for_sort = roughness_scores(:);
    roughness_for_sort(~isfinite(roughness_for_sort)) = inf;
    [~, order] = sortrows([roughness_for_sort, original_positions], [1, 2]);

    Psi_candidate = Psi_candidate(:, order);
    remove_mean_flags = remove_mean_flags(order);

    % Orthonormalize Psi
    Psi = stabilize_masked_basis_ordered( ...
        Psi_candidate, valid_fluid_mask, remove_mean_flags);
end

function Psi = build_sinx_siny_modes(nx, ny, valid_fluid_mask, n_modes)
    if n_modes <= 0
        Psi = zeros(nx * ny, 0);
        return;
    end

    pair_table = generate_low_order_wave_pairs(n_modes);
    x_hat = ((1:nx) - 0.5) / nx;
    y_hat = ((1:ny) - 0.5) / ny;

    Psi = zeros(nx * ny, size(pair_table, 1));
    for mode_index = 1:size(pair_table, 1)
        mx = pair_table(mode_index, 1);
        my = pair_table(mode_index, 2);

        x_mode = sin(pi * mx * x_hat);
        y_mode = sin(pi * my * y_hat);
        mode_2d = y_mode(:) * x_mode(:).';
        mode_2d(~valid_fluid_mask) = 0;

        Psi(:, mode_index) = mode_2d(:);
    end
end

function pair_table = generate_low_order_wave_pairs(n_pairs)
    if n_pairs <= 0
        pair_table = zeros(0, 2);
        return;
    end

    max_order = 1;
    while true
        candidate_pairs = enumerate_diagonal_wave_pairs(max_order);
        if size(candidate_pairs, 1) >= n_pairs
            break;
        end
        max_order = max_order + 1;
    end

    max_order = max_order + 4;
    all_pairs = enumerate_diagonal_wave_pairs(max_order);
    pair_table = all_pairs(1:n_pairs, :);
end

function pair_table = enumerate_diagonal_wave_pairs(max_order)
    pair_table = zeros(max_order * max_order, 2);
    count = 0;

    % Diagonal ordering:
    % (1,1),
    % (1,2),(2,1),
    % (1,3),(2,2),(3,1), ...
    for diagonal_sum = 2:(2 * max_order)
        mx_min = max(1, diagonal_sum - max_order);
        mx_max = min(max_order, diagonal_sum - 1);

        for mx = mx_min:mx_max
            my = diagonal_sum - mx;
            if my < 1 || my > max_order
                continue;
            end
            count = count + 1;
            pair_table(count, :) = [mx, my];
        end
    end

    pair_table = pair_table(1:count, :);
end

function [Psi, eigenvalues] = build_graph_laplacian_modes(nx, ny, valid_fluid_mask, n_modes)
    if n_modes <= 0
        Psi = zeros(nx * ny, 0);
        eigenvalues = [];
        return;
    end

    valid_linear_mask = valid_fluid_mask(:);
    valid_indices = find(valid_linear_mask);
    n_valid = numel(valid_indices);
    n_modes = min(max(round(n_modes), 1), n_valid);

    node_id_grid = zeros(ny, nx);
    node_id_grid(valid_fluid_mask) = 1:n_valid;

    estimated_edges = 2 * n_valid;
    row_idx = zeros(estimated_edges, 1);
    col_idx = zeros(estimated_edges, 1);
    edge_count = 0;

    for iy = 1:ny
        for ix = 1:nx
            if ~valid_fluid_mask(iy, ix)
                continue;
            end

            current_id = node_id_grid(iy, ix);

            if ix < nx && valid_fluid_mask(iy, ix + 1)
                edge_count = edge_count + 1;
                row_idx(edge_count) = current_id;
                col_idx(edge_count) = node_id_grid(iy, ix + 1);
            end

            if iy < ny && valid_fluid_mask(iy + 1, ix)
                edge_count = edge_count + 1;
                row_idx(edge_count) = current_id;
                col_idx(edge_count) = node_id_grid(iy + 1, ix);
            end
        end
    end

    row_idx = row_idx(1:edge_count);
    col_idx = col_idx(1:edge_count);

    adjacency = sparse([row_idx; col_idx], [col_idx; row_idx], 1, n_valid, n_valid);
    degree_vector = full(sum(adjacency, 2));
    laplacian_matrix = spdiags(degree_vector, 0, n_valid, n_valid) - adjacency;
    laplacian_matrix = 0.5 * (laplacian_matrix + laplacian_matrix.');

    if n_valid == 1
        eigenvectors_valid = 1;
        eigenvalues = 0;
    else
        try
            opts = struct();
            opts.isreal = true;
            opts.tol = 1e-10;
            opts.maxit = 5e3;
            [eigenvectors_valid, eigenvalue_matrix] = eigs(laplacian_matrix, n_modes, 'sa', opts);
            eigenvalues = real(diag(eigenvalue_matrix));
        catch
            [eigenvectors_all, eigenvalue_vector] = eig(full(laplacian_matrix), 'vector');
            [eigenvalues, order] = sort(real(eigenvalue_vector(:)), 'ascend');
            eigenvectors_valid = real(eigenvectors_all(:, order(1:n_modes)));
            eigenvalues = eigenvalues(1:n_modes);
        end
    end

    [eigenvalues, order] = sort(real(eigenvalues(:)), 'ascend');
    eigenvectors_valid = real(eigenvectors_valid(:, order));

    for mode_index = 1:size(eigenvectors_valid, 2)
        v_norm = norm(eigenvectors_valid(:, mode_index), 2);
        if v_norm > 0
            eigenvectors_valid(:, mode_index) = eigenvectors_valid(:, mode_index) / v_norm;
        end
    end
    [eigenvectors_valid, ~] = qr(eigenvectors_valid, 0);

    Psi = zeros(nx * ny, size(eigenvectors_valid, 2));
    for mode_index = 1:size(eigenvectors_valid, 2)
        mode_vector = zeros(nx * ny, 1);
        mode_vector(valid_linear_mask) = eigenvectors_valid(:, mode_index);
        Psi(:, mode_index) = mode_vector;
    end
end

function [Psi_stable, kept_mode_indices] = stabilize_masked_basis_ordered(Psi_candidate, valid_fluid_mask, remove_mean_per_mode)
    [m, n_modes] = size(Psi_candidate);
    valid_linear_mask = valid_fluid_mask(:);

    if nargin < 3 || isempty(remove_mean_per_mode)
        remove_mean_per_mode = true(1, n_modes);
    end

    if isscalar(remove_mean_per_mode)
        remove_mean_per_mode = repmat(logical(remove_mean_per_mode), 1, n_modes);
    else
        remove_mean_per_mode = logical(remove_mean_per_mode(:).');
    end

    tiny_norm_tol = 1e-10;
    dependent_tol = 1e-8;

    Psi_clean = zeros(m, n_modes);
    for j = 1:n_modes
        v = Psi_candidate(:, j);
        v(~valid_linear_mask) = 0;
        v(~isfinite(v)) = 0;

        if remove_mean_per_mode(j)
            mean_valid = mean(v(valid_linear_mask));
            v(valid_linear_mask) = v(valid_linear_mask) - mean_valid;
        end

        Psi_clean(:, j) = v;
    end

    valid_norms = vecnorm(Psi_clean(valid_linear_mask, :), 2, 1);
    candidate_indices = find(isfinite(valid_norms) & valid_norms > tiny_norm_tol);

    if isempty(candidate_indices)
        error('Secondary library is empty after orthogonalization.');
    end

    [Q_valid, R_valid] = qr(Psi_clean(valid_linear_mask, candidate_indices), 0);

    if isempty(R_valid)
        rank_qr = 0;
    else
        diag_R = abs(diag(R_valid));
        rank_qr = sum(diag_R > dependent_tol * max(1, diag_R(1)));
    end

    if rank_qr < 1
        error('Secondary library is empty after orthogonalization.');
    end

    Q_valid = Q_valid(:, 1:rank_qr);
    Psi_stable = zeros(m, rank_qr);
    Psi_stable(valid_linear_mask, :) = Q_valid;
    kept_mode_indices = candidate_indices(1:rank_qr);
end

function roughness_scores = compute_candidate_graph_roughness_scores( ...
    Psi_candidate, valid_fluid_mask, remove_mean_flags, nx, ny)

    [~, n_modes] = size(Psi_candidate);
    valid_linear_mask = valid_fluid_mask(:);

    if isscalar(remove_mean_flags)
        remove_mean_flags = repmat(logical(remove_mean_flags), 1, n_modes);
    else
        remove_mean_flags = logical(remove_mean_flags(:).');
    end

    [edge_first, edge_second] = build_valid_grid_neighbor_pairs( ...
        reshape(valid_fluid_mask, ny, nx));

    roughness_scores = inf(1, n_modes);
    tiny_norm_tol = 1e-10;

    for j = 1:n_modes
        v = clean_candidate_mode_for_basis( ...
            Psi_candidate(:, j), valid_linear_mask, remove_mean_flags(j));

        denominator = sum(abs(v(valid_linear_mask)).^2);
        if ~isfinite(denominator) || denominator <= tiny_norm_tol^2
            continue;
        end

        if isempty(edge_first)
            numerator = 0;
        else
            edge_differences = v(edge_first) - v(edge_second);
            numerator = sum(abs(edge_differences).^2);
        end

        roughness_scores(j) = real(numerator / denominator);
    end

    roughness_scores(roughness_scores < 0 & roughness_scores > -1e-12) = 0;
end

function v = clean_candidate_mode_for_basis(v, valid_linear_mask, remove_mean)
    v = double(v(:));
    v(~valid_linear_mask) = 0;
    v(~isfinite(v)) = 0;

    if remove_mean
        valid_values = v(valid_linear_mask);
        if ~isempty(valid_values)
            v(valid_linear_mask) = valid_values - mean(valid_values);
        end
    end
end

function [edge_first, edge_second] = build_valid_grid_neighbor_pairs(valid_grid)
    [ny, nx] = size(valid_grid);
    n_valid = nnz(valid_grid);
    maximum_edges = max(1, 2 * n_valid);

    edge_first = zeros(maximum_edges, 1);
    edge_second = zeros(maximum_edges, 1);
    edge_count = 0;

    for ix = 1:nx
        for iy = 1:ny
            if ~valid_grid(iy, ix)
                continue;
            end

            current_index = sub2ind([ny, nx], iy, ix);

            if ix < nx && valid_grid(iy, ix + 1)
                edge_count = edge_count + 1;
                edge_first(edge_count) = current_index;
                edge_second(edge_count) = sub2ind([ny, nx], iy, ix + 1);
            end

            if iy < ny && valid_grid(iy + 1, ix)
                edge_count = edge_count + 1;
                edge_first(edge_count) = current_index;
                edge_second(edge_count) = sub2ind([ny, nx], iy + 1, ix);
            end
        end
    end

    edge_first = edge_first(1:edge_count);
    edge_second = edge_second(1:edge_count);
end

function active_rank = choose_active_rank(singular_values, requested_rank)
    if isempty(singular_values) || max(singular_values) <= 0
        active_rank = 1;
        return;
    end
    tolerance = 1e-10 * max(singular_values);
    physical_rank = sum(singular_values > tolerance);
    physical_rank = max(physical_rank, 1);
    active_rank = min(round(requested_rank), physical_rank);
    active_rank = max(active_rank, 1);
end

function B = solve_right_unregularized(RHS, coefficient_matrix)
    coefficient_matrix = 0.5 * (coefficient_matrix + coefficient_matrix.');

    if rcond(full(coefficient_matrix)) > eps(max(size(coefficient_matrix)))
        B = RHS / coefficient_matrix;
        return;
    end

    B = RHS * pinv(full(coefficient_matrix));
end

function [L, S, iter, info] = rpca_ialm(D, rpca_lambda, tolerance, maximum_iterations)
    [m, n] = size(D);
    L = zeros(m, n);
    S = zeros(m, n);

    rpca_lambda_input = rpca_lambda;
    rpca_lambda_effective = rpca_lambda / sqrt(max(m, n));

    norm_D_fro = norm(D, 'fro');
    norm_D_two = norm(D, 2);
    norm_D_inf = norm(D(:), inf) / max(rpca_lambda_effective, eps);
    dual_norm = max(norm_D_two, norm_D_inf);
    Y = D / max(dual_norm, eps);

    mu = 1.25 / max(norm_D_two, eps);
    mu_max = mu * 1e7;
    rho = 1.5;

    converged = false;
    final_relative_residual = inf;

    for iter = 1:maximum_iterations
        S = soft_threshold(D - L + (1 / mu) * Y, rpca_lambda_effective / mu);

        [U, Sigma, V] = svd(D - S + (1 / mu) * Y, 'econ');
        sigma = diag(Sigma);
        active = sigma > (1 / mu);

        if any(active)
            sigma_thresholded = sigma(active) - (1 / mu);
            L = U(:, active) * diag(sigma_thresholded) * V(:, active)';
        else
            L = zeros(m, n);
        end

        residual = D - L - S;
        Y = Y + mu * residual;
        mu = min(rho * mu, mu_max);

        final_relative_residual = norm(residual, 'fro') / max(norm_D_fro, eps);
        if final_relative_residual < tolerance
            converged = true;
            break;
        end
    end

    info = struct();
    info.converged = converged;
    info.final_relative_residual = final_relative_residual;
    info.rpca_lambda_input = rpca_lambda_input;
    info.rpca_lambda_effective = rpca_lambda_effective;
end

function X = soft_threshold(X, threshold)
    X = sign(X) .* max(abs(X) - threshold, 0);
end

function X_full = expand_to_full_grid(X_valid, valid_fluid_mask)
    X_full = zeros(numel(valid_fluid_mask), size(X_valid, 2));
    X_full(valid_fluid_mask(:), :) = X_valid;
end
