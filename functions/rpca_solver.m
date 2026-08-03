% rpca_solver.m
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

function [L, S, info] = inexact_alm_rpca( ...
        X, lambda_scale, ...
        convergence_tolerance, maximum_iterations)
% Solve PCP using inexact augmented Lagrangian updates.
%
%   minimize ||L||_* + lambda_0 ||S||_1
%   subject to X = L + S
%
% with
%
%   lambda_0 = lambda_scale / sqrt(max(m,n)).

    [m, n] = size(X);

    lambda_0 = lambda_scale / sqrt(max(m, n));

    X_norm = norm(X, 'fro');
    spectral_norm = norm(X, 2);

    dual_norm = max( ...
        spectral_norm, ...
        norm(X(:), inf) / lambda_0);

    Y = X / dual_norm;

    L = zeros(size(X));
    S = zeros(size(X));

    mu = 1.25 / spectral_norm;
    mu_maximum = mu * 1e7;
    mu_growth_factor = 1.5;

    stop_reason = 'maximum_iterations';

    for iteration = 1:maximum_iterations

        % Update S

        target_S = X - L + Y / mu;

        S = soft_threshold( ...
            target_S, ...
            lambda_0 / mu);

        % Update L

        target_L = X - S + Y / mu;

        [U, Sigma, V] = svd(target_L, 'econ');

        singular_values = diag(Sigma);
        singular_value_threshold = 1 / mu;

        retained_indices = ...
            singular_values > singular_value_threshold;

        if any(retained_indices)

            thresholded_values = ...
                singular_values(retained_indices) ...
                - singular_value_threshold;

            L = U(:, retained_indices) ...
                * diag(thresholded_values) ...
                * V(:, retained_indices)';

        else

            L = zeros(size(X));

        end

        % Update Y

        primal_residual = X - L - S;

        Y = Y + mu * primal_residual;

        relative_primal_residual = ...
            norm(primal_residual, 'fro') / X_norm;

        if relative_primal_residual < convergence_tolerance

            stop_reason = 'converged';
            break;

        end

        % Update mu

        mu = min( ...
            mu_growth_factor * mu, ...
            mu_maximum);

    end

    info.iterations = iteration;
    info.stop_reason = stop_reason;

    info.relative_primal_residual = ...
        relative_primal_residual;

    info.lambda_0 = lambda_0;
    info.recovered_rank = rank(L);

end

function Z = soft_threshold(X, threshold)

    Z = sign(X) .* max(abs(X) - threshold, 0);

end
