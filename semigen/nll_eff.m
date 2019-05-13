function nll = nll_eff(X_cau, Y, X_eff, theta)
% This function computes the negative log likelihood of the parameters 
% given the effect features in X_eff, the corresponding binary label in 
% {0,1}, and the causal features X_cau. The likelihood is assumed to be
% linear Gaussian with mean X*theta_i and std. sigma_i.
[n,p] = size(X_cau);
X = [ones(n,1), X_cau];
theta_0 = theta(1:p+1); sigma_0 = exp(theta(p+2));
theta_1 = theta(p+3:2*p+3); sigma_1 = exp(theta(2*p+4));

nll = Y .* ((X_eff - X*theta_1).^2/(2*sigma_1^2) + 0.5 * log(sigma_1^2)) + ...
    (1-Y) .* ((X_eff - X*theta_0).^2/(2*sigma_0^2) + 0.5 * log(sigma_0^2));
end

