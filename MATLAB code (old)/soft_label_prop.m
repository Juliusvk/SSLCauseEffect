function Y_u = soft_label_prop( X_lC, X_lE, Y_l, X_uC, X_uE, lam )
%This function implements a version of label propagation with soft labels
%using weighted regression to recompute the fits of p(X_E|X_C,Y).

[n_l, p_l] = size(X_lC);
[n_u, p] = size(X_uC);
assert(isequal(p_l,p))
n = n_l + n_u;
Y_l = Y_l-1; % Y has labels in [0,1]

% compute initial fits and soft labels
b_0 = ridge(X_lE(Y_l==0), X_lC(Y_l==0,:), lam, 0);
b_1 = ridge(X_lE(Y_l==1), X_lC(Y_l==1,:), lam, 0);
assert(isequal(size(b_0),size(b_1)))

% unbiased estimates or error variances of regression lines
n_0 = sum(Y_l==0); n_1 = sum(Y_l==1);
s_0 = sum((X_lE(Y_l==0)-[ones(n_0,1),X_lC(Y_l==0,:)]*b_0).^2)/(n_0-p-1);
s_1 = sum((X_lE(Y_l==1)-[ones(n_1,1),X_lC(Y_l==1,:)]*b_1).^2)/(n_1-p-1);
s_0 = sqrt(s_0); s_1 = sqrt(s_1); %transform to standard deviations

X_E_pred = [ones(n_u,1), X_uC] * [b_0, b_1];
p0 = normpdf(X_uE, X_E_pred(:,1), s_0);
p1 = normpdf(X_uE, X_E_pred(:,2), s_1);

% assign initial soft labels
Y_u = p1./(p0+p1);

% define matrices for weighted regression
X_C0 = [ones(n_0+n_u,1), [X_lC(Y_l==0,:);X_uC]]; 
X_E0 = [X_lE(Y_l==0,:);X_uE];
X_C1 = [ones(n_1+n_u,1), [X_lC(Y_l==1,:);X_uC]];
X_E1 = [X_lE(Y_l==1,:);X_uE];

converged = 0;
while ~converged
    %perform weighted regression
    W_0 = diag(1 - [Y_l(Y_l==0);Y_u]);
    W_1 = diag([Y_l(Y_l==1);Y_u]);
    b_0 = (X_C0'*W_0*X_C0 + lam * eye(p+1)) \ X_C0' * W_0 * X_E0;
    b_1 = (X_C1'*W_1*X_C1 + lam * eye(p+1)) \ X_C1' * W_1 * X_E1;
    assert(isequal(size(b_0), size(b_1)));
   
    % unbiased estimates or error variances of regression lines
    s_0 = sum((X_E0-X_C0*b_0).^2)/(n_0+n_u-p-1);
    s_1 = sum((X_E1-X_C1*b_1).^2)/(n_1+n_u-p-1);
    
    % compute new soft labels
    X_E_pred = [ones(n_u,1), X_uC] * [b_0, b_1];
    p0 = normpdf(X_uE, X_E_pred(:,1), s_0);
    p1 = normpdf(X_uE, X_E_pred(:,2), s_1);
    Y_u_new = p1./(p0+p1);
    
    % check for convergence
    if max((Y_u_new-Y_u).^2) < 1e-3
        converged = 1;
    end
    Y_u = Y_u_new;
end

end

