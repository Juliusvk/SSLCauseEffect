close all; clc; clear all;

%% load and preprocess data
dataset = 'diabetes';
switch dataset
    case 'diabetes'
        T = readtable('diabetes.csv'); A = table2array(T);
        X = A(:,1:8); Y = A(:,9)+1;
        %impute missing glucose and BMI values by class mean instead of 0
        X(X(:,2)==0,2) = (Y(X(:,2)==0)-1) * mean(X(and(Y==2,X(:,2)~=0), 2)) + ...
            (2-Y(X(:,2)==0)) * mean(X(and(Y==1, X(:,2)~=0), 2));
        X(X(:,6)==0,6) = (Y(X(:,6)==0)-1) * mean(X(and(Y==2,X(:,6)~=0), 6)) + ...
            (2-Y(X(:,6)==0)) * mean(X(and(Y==1, X(:,6)~=0), 6));
        % Feature selection: Pregnancy, BMI, PedigreeFunc, Glucose (p < 0.01)
        X = A(:, [6, 2]);
        % Feature selection: BMI, Glucose (p < 0.01)
        % X = A(:, [6, 2]);
    case 'heartdisease'
        
end

% Standardize Data
X = (X - mean(X))./std(X);

%% Perform Experiment
n_iter = 100; n_train = 768; n_test = 0;
n_labelled = 2^5; n_unlabelled_range = 2.^(5:9);
lam_ridge = 1;
% threshold = 10;
% lam_semigen = 0.1;
% options = optimset('MaxFunEvals',1e5, 'MaxIter', 1e5);

for iter = 1:n_iter
tic
%% separate into train and test data
[X_tr, Y_tr, X_te, Y_te] = train_test_split(X, Y, n_train, n_test);

for i = 1:length(n_unlabelled_range)
    
    %% split into labelled and unlabelled
    n_unlabelled = n_unlabelled_range(i);
    [X_lab, Y_lab, X_unl, Y_unl] = train_test_split(X_tr, Y_tr,...
        n_labelled, n_unlabelled);
    switch dataset
        case 'diabetes'     
            X_lab_cau = X_lab(:, 1:end-1); X_lab_eff = X_lab(:, end);
            X_unl_cau = X_unl(:, 1:end-1); X_unl_eff = X_unl(:, end);
        case 'heartdisease'
    end   
    
    %% Baseline 1: logistic regression on labelled data only
    [B_base,~,~] = mnrfit(X_lab, Y_lab);
    [transd.AUC_base(i, iter), ~,~] =  evaluate(B_base, X_unl, Y_unl);
    %[res.AUC_base(i, iter), ~, ~] = evaluate(B_base, X_te, Y_te);

    %% Hard label propagation
    Y_hard = hard_label_prop(X_lab_cau, X_lab_eff, Y_lab, X_unl_cau, X_unl_eff, lam_ridge);
    [~,~,~,transd.AUC_hardlp(i,iter)] = perfcurve(Y_unl, Y_hard-1, 2);
    
    [B_hardlp,~,~] = mnrfit([X_lab; X_unl], [Y_lab; Y_hard]);
    [transd.AUC_hardlp_LR(i, iter), ~,~] =  evaluate(B_hardlp, X_unl, Y_unl);

    %% Soft label propagation
    Y_soft = soft_label_prop(X_lab_cau, X_lab_eff, Y_lab, X_unl_cau, X_unl_eff, lam_ridge);
    [~,~,~,transd.AUC_softlp(i,iter)] = perfcurve(Y_unl, Y_soft, 2);

    
    %     [X_lab_cau, X_lab_eff,Y_lab, X_unl_cau, X_unl_eff] = lin_fun_label_prop(...
    %         X_lab_cau, X_lab_eff, Y_lab, X_unl_cau, X_unl_eff, lam_ridge, threshold);
    %     [B_clp,~,~] = mnrfit([X_lab_cau,X_lab_eff], Y_lab);
    %     [res.AUC_clp(i, iter), ~, ~] = evaluate(B_clp, X_unl, Y_unl);
    
    %% Semi-Generative Model
    %     fun = @(theta) nll_pooled(X_lab_cau,X_lab_eff,Y_lab,...
    %         X_unl_cau,X_unl_eff,theta,lam_semigen);
    %     [~,p_cau] = size(X_lab_cau);
    %     th_0 = zeros(3*p_cau+5,1);
    %     th_0 = 0.1 * randn(3*p_cau+5,1);
    %     theta = fminsearch(fun,th_0, options);
    
end
toc
end

%%
AUCs = figure(2); hold on; xlabel('log_2(unlabelled examples)'); ylabel('AUROC');
errorbar(log2(n_unlabelled_range), mean(transd.AUC_base,2), std(transd.AUC_base'), 'LineWidth', 2);
errorbar(log2(n_unlabelled_range), mean(transd.AUC_hardlp,2), std(transd.AUC_hardlp'), 'LineWidth', 2);
errorbar(log2(n_unlabelled_range), mean(transd.AUC_hardlp_LR,2), std(transd.AUC_hardlp_LR'), 'LineWidth', 2);
errorbar(log2(n_unlabelled_range), mean(transd.AUC_softlp,2), std(transd.AUC_softlp'), 'LineWidth', 2);
legend('linear log. reg.', 'hard label prop.', 'hard label prop + log. reg.','soft label prop');
title(sprintf('Transductive, %i labelled examples', n_labelled))




%% obsolete code

% train and test on full dataset: oracle
%         [B_full,~,~] = mnrfit(X, Y);
%         [AUC_full, x_roc_full, y_roc_full] = evaluate(B_full, X, Y);
% model_comparison = figure(1); hold on;
% xlabel('False Positive Rate'); ylabel('True Positive Rate');
% plot([0 1],[0 1], 'DisplayName', 'random','LineWidth', 2);
% txt = sprintf('%i labelled, AUC = %.3f', n_labelled, res.AUC(i));
% hold off; legend show;

% plot(x_roc,y_roc, 'DisplayName', txt, 'LineWidth', 2);
    
