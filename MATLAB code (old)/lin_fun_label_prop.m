function [X_lab_cau, X_lab_eff,Y_lab, X_unl_cau, X_unl_eff] = lin_fun_label_prop(...
    X_lab_cau, X_lab_eff, Y_lab, X_unl_cau, X_unl_eff, lam_ridge, threshold)

% This function implements a linear version of function-based label
% propagation: it uses the current labelled sample to fit two linear models
% from causes to effects and then newly labels those unlabelled points
% which most agree with these fits. This is repeated until all points are
% labelled or until all remaining unlabelled points can be equally well
% (within some threshold) explained by both linear fits.

while size(X_unl_eff, 1) > 0
    % regress effects on causes on labelled sample and predict
    b_1 = ridge(X_lab_eff(Y_lab==1), X_lab_cau(Y_lab==1,:), lam_ridge, 0);
    b_2 = ridge(X_lab_eff(Y_lab==2), X_lab_cau(Y_lab==2,:), lam_ridge, 0);
    assert(isequal(size(b_1),size(b_2)))
    X_unl_eff_pred = [ones(size(X_unl_cau,1),1), X_unl_cau] * [b_1, b_2];
    diff = (X_unl_eff - X_unl_eff_pred).^2;
    ratio = diff(:,1)./diff(:,2);
    
    % newly label the best fitting unlabelled points
    [min_val, min_idx] = min(ratio);
    [max_val, max_idx] = max(ratio);
    del_min = 0; del_max = 0;
    if min_val < 1/threshold
        del_min = 1; Y_lab = [Y_lab; 1];
        X_lab_cau = [X_lab_cau; X_unl_cau(min_idx,:)];
        X_lab_eff = [X_lab_eff; X_unl_eff(min_idx,:)];     
    end
    if max_val > threshold
        del_max = 1; Y_lab = [Y_lab; 2];
        X_lab_cau = [X_lab_cau; X_unl_cau(max_idx,:)];
        X_lab_eff = [X_lab_eff; X_unl_eff(max_idx,:)];     
    end
    
    % delete newly labelled from unlabelled sample
    if del_min && del_max
        X_unl_cau([min_idx, max_idx],:) = [];
        X_unl_eff([min_idx, max_idx],:) = [];
    elseif del_min
        X_unl_cau(min_idx,:) = [];
        X_unl_eff(min_idx,:) = [];
    elseif del_max
        X_unl_cau(max_idx,:) = [];
        X_unl_eff(max_idx,:) = [];
    end
    
    % when no points have been newly labelled, stop
    if min_val > 1/threshold && max_val < threshold
        break
    end
end

end
