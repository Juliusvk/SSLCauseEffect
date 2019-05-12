function [X_tr, Y_tr, X_te, Y_te] = train_test_split(X, Y, n_tr, n_te)
% Splits the data into n_tr training and n_te test instances.
n = size(X,1);
assert(n == size(Y,1));
assert(n >= n_tr + n_te);
idx_tr = randi(n, [n_tr,1]);
X_tr = X(idx_tr,:);
Y_tr = Y(idx_tr,:);
X(idx_tr,:) = [];
Y(idx_tr,:) = [];
if n - n_tr > 0
    idx_te = randi(n - n_tr, [n_te,1]);
    X_te = X(idx_te,:);
    Y_te = Y(idx_te,:);
else
    X_te = [];
    Y_te = [];
end
end

