
% Example sparse matrix R
R = [
    5 3 0 1;
    4 0 0 1;
    1 1 0 5;
    1 0 0 4;
    0 1 5 4;
];

% Matrix factorization parameters
k = 2;
lambda = 0.1;
n_epochs = 100;
learning_rate = 0.01;

% Perform matrix factorization
rng(0);
[P, Q] = matrix_factorization_SGD(R, k, lambda, n_epochs, learning_rate);

% Compute the completed matrix
R_completed = P * Q';

% Alec's sanity check
S = R > 0;
err = sum((S.*(R - R_completed)).^2,'all')


% Perform matrix factorization
rng(0);
[P, Q] = matrix_factorization_SGD_vectorized(R, k, lambda, n_epochs, learning_rate);

% Compute the completed matrix
R_completed = P * Q';

% Alec's sanity check
S = R > 0;
err = sum((S.*(R - R_completed)).^2,'all')

