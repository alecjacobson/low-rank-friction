function [P, Q] = matrix_factorization_sgd(R, k, lambda, n_epochs, learning_rate)
    [m, n] = size(R);
    P = rand(m, k);
    Q = rand(n, k);

    % Find the observed entries (non-zero) in R
    [i_obs, j_obs, r_obs] = find(R);
    n_obs = length(r_obs);

    % Perform Stochastic Gradient Descent (SGD)
    for epoch = 1:n_epochs
        % Shuffle the observed entries
        idx = randperm(n_obs);

        for idx_i = 1:n_obs
            i = i_obs(idx(idx_i));
            j = j_obs(idx(idx_i));
            rij = r_obs(idx(idx_i));

            % Compute the error for the current entry
            eij = rij - P(i, :) * Q(j, :)';

            % Update P and Q using the computed gradient and learning rate
            P(i, :) = P(i, :) + learning_rate * (2 * eij * Q(j, :) - lambda * P(i, :));
            Q(j, :) = Q(j, :) + learning_rate * (2 * eij * P(i, :) - lambda * Q(j, :));
        end

        % Compute the total error for this epoch
        total_error = 0;
        for idx_i = 1:n_obs
            i = i_obs(idx(idx_i));
            j = j_obs(idx(idx_i));
            rij = r_obs(idx(idx_i));
            total_error = total_error + (rij - P(i, :) * Q(j, :)')^2;
        end
        fprintf('Epoch: %d, Total Error: %f\n', epoch, total_error);
    end
end
