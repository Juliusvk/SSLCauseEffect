function [X_tr, Y_tr, X_te, Y_te] = train_test_split(X, Y, n_tr, n_te)
% Splits the data into n_tr training and n_te test instances.
n = size(X,1);
assert(n == size(Y,1));
assert(n >= n_tr + n_te);

sufficient_labels = 0;
while ~sufficient_labels
    idx_tr = randi(n, [n_tr,1]);
    Y_tr = Y(idx_tr,:);
    if sum(Y_tr==1) > 1 && sum(Y_tr==2) > 1
        sufficient_labels = 1;
        X_tr = X(idx_tr,:);
        X(idx_tr,:) = []; Y(idx_tr,:) = [];
        if n - n_tr > 0
            idx_te = randi(n - n_tr, [n_te,1]);
            X_te = X(idx_te,:); Y_te = Y(idx_te,:);
        else
            X_te = []; Y_te = [];
        end
    end
end
end

