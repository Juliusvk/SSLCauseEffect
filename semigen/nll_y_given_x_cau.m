function nll = nll_y_given_x_cau(X_cau, Y, theta)
% This function computes the negative log-likelihood (nll) of theta given
% the labels {0,1} in Y and the causal features X_cau, using a linear 
% logistic regression model.
X = [ones(size(X_cau,1),1), X_cau];
eta = X * theta;
nll = log(1+exp(-eta)) + (1-Y).* eta;
end

