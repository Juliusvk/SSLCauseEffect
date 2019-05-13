function nll = nll_pooled(X_lab_cau,X_lab_eff,Y_lab,X_unl_cau,X_unl_eff,theta,lam)
%This function computes the pooled negative loglikelihood as a combination 
%of the labelled data NLL and lam times the unlabelled data NLL.
[~,p] = size(X_lab_cau);
nll = sum( nll_y_given_x_cau(X_lab_cau, Y_lab, theta(1:p+1)) + ...
    nll_eff(X_lab_cau, Y_lab, X_lab_eff, theta(p+2:3*p+5)) ) + lam * ...
    sum( nll_unlabelled(X_unl_cau, X_unl_eff, theta) );
end

