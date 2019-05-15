function Y_u = hard_label_prop( X_lC, X_lE, Y_l, X_uC, X_uE, lam)
%This function implements a form of label propagation with hard labels
%taking into account the causal structure of the problem. Labels are 1
%(class0), 2 (class1), and 0 (unlabelled).

[n_u, p] = size(X_uC);
Y = [Y_l; zeros(n_u,1)];
X_C = [X_lC; X_uC];
X_E = [X_lE; X_uE];

while sum(Y == 0) > 0
    % regress effects on causes on labelled sample and predict
    b_1 = ridge(X_E(Y==1), X_C(Y==1,:), lam, 0);
    b_2 = ridge(X_E(Y==2), X_C(Y==2,:), lam, 0);
    assert(isequal(size(b_1),size(b_2)))
    
    n_1 = sum(Y==1); 
    n_2 = sum(Y==2);
    n_0 = sum(Y==0);

    % unbiased estimates or error variances of regression lines
    s_1 = sum((X_E(Y==1)-[ones(n_1,1),X_C(Y==1,:)]*b_1).^2)/(n_1-p-1);
    s_2 = sum((X_E(Y==2)-[ones(n_2,1),X_C(Y==2,:)]*b_2).^2)/(n_2-p-1);

    X_E_pred = [ones(n_0,1), X_C(Y==0,:)] * [b_1, b_2];
    p1 = normpdf(X_E(Y==0), X_E_pred(:,1), s_1);
    p2 = normpdf(X_E(Y==0), X_E_pred(:,2), s_2);
    
    % THE NEXT STEP IS SOMEWHAT QUESTIONABLE SINCE IT DOES NOT TAKE P(Y|X_C)
    % INTO ACCOUNT AND IS THEREFORE NOT QUITE CORRECT. COULD FIT A STRONGLY
    % REGULARISED LOG.REG. TO ESTIMATE P(Y|X_C) INITIALLY.
    
    % transform likelihoods p(X_E|X_C,Y=1), p(X_E|X_C,Y=2) into weights.
    w1 = zeros(size(Y));
    w1(Y==0) = p1./(p1+p2);
    w2 = zeros(size(Y));
    w2(Y==0) = p2./(p1+p2); 
    
    % newly label the best fitting unlabelled points
    [max_val1, max_idx1] = max(w1);
    [max_val2, max_idx2] = max(w2);
    if max_val1 > max_val2
        Y(max_idx1) = 1;
    else
        Y(max_idx2) = 2;
    end
end
Y_u = Y(end-n_u+1:end);
end
