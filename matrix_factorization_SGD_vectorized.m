function [P, Q] = matrix_factorization_SGD_vectorized(R, k, lambda, n_epochs, learning_rate)
    [m, n] = size(R);
    P = rand(m, k);
    Q = rand(n, k);

    % Find the observed entries (non-zero) in R
    [i_obs, j_obs, r_obs] = find(R);
    n_obs = length(r_obs);
    idx_obs = sub2ind(size(R), i_obs, j_obs);

    % Perform Stochastic Gradient Descent (SGD)
    for epoch = 1:n_epochs
        % Shuffle the observed entries
        idx = randperm(n_obs);
        i_obs = i_obs(idx);
        j_obs = j_obs(idx);
        r_obs = r_obs(idx);
        
        % Compute all errors at once (vectorized)
        E = R - P * Q';
        E(idx_obs) = E(idx_obs) - r_obs;
        
        % Update P and Q using the computed gradient and learning rate
        P = P - learning_rate * (-2 * E * Q + lambda * P);
        Q = Q - learning_rate * (-2 * E' * P + lambda * Q);

        % Compute the total error for this epoch
        total_error = sum((R(idx_obs) - P(i_obs, :) * Q(j_obs, :)').^2);
        fprintf('Epoch: %d, Total Error: %f\n', epoch, total_error);
    end
end
