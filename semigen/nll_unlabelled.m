function nll = nll_unlabelled( X_cau, X_eff, theta )
%This function computes the likelihood of theta given only the unlabelled
%data.
X = [ones(size(X_cau,1),1), X_cau];
[~,p] = size(X_cau);
eta = X * theta(1:p+1);
th = theta(p+2:end);
theta_0 = th(1:p+1); sigma_0 = exp(th(p+2));
theta_1 = th(p+3:2*p+3); sigma_1 = exp(th(2*p+4));

nll = log(1+exp(-eta)) - log( normpdf(X_eff, X * theta_1, sigma_1) + ...
    exp(-eta) .* normpdf(X_eff, X * theta_0, sigma_0));

end

