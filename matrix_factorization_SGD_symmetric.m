function [P] = matrix_factorization_SGD_symmetric(R, k, lambda, n_epochs, learning_rate)
    [m, n] = size(R);
    assert(m == n, 'Matrix R must be symmetric.');

    P = rand(m, k);

    % Find the observed entries (non-zero) in R
    [i_obs, j_obs, r_obs] = find(R);
    n_obs = length(r_obs);

    % Perform Stochastic Gradient Descent (SGD)
    for epoch = 1:n_epochs
        % Shuffle the observed entries
        idx = randperm(n_obs);
        i_obs = i_obs(idx);
        j_obs = j_obs(idx);
        r_obs = r_obs(idx);

        for idx_i = 1:n_obs
            i = i_obs(idx_i);
            j = j_obs(idx_i);

            % Compute the error for the current entry
            eij = r_obs(idx_i) - P(i, :) * P(j, :)';

            % Update P using the computed gradient and learning rate
            P(i, :) = P(i, :) + learning_rate * (2 * eij * P(j, :) - lambda * P(i, :));
            P(j, :) = P(j, :) + learning_rate * (2 * eij * P(i, :) - lambda * P(j, :));
        end

        % Compute the total error for this epoch
        total_error = sum((R(sub2ind(size(R), i_obs, j_obs)) - sum(P(i_obs, :) .* P(j_obs, :), 2)).^2);
        fprintf('Epoch: %d, Total Error: %f\n', epoch, total_error);
    end
end

