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
        X = A(:, [1, 6, 7, 2]);
        % Feature selection: BMI, Glucose (p < 0.01)
        % X = A(:, [6, 2]);
    case 'heartdisease'
        
end

% Standardize Data
X = (X - mean(X))./std(X);

%% Perform Experiment
n_iter = 100; n_train = 512; n_test = 256;
n_labelled = 2^5; n_unlabelled_range = 2.^(0:8);
lam_ridge = 1; threshold = 10; lam_semigen = 0.1;
options = optimset('MaxFunEvals',1e5, 'MaxIter', 1e5);

for iter = 1:n_iter
tic
%% separate into train and test data
[X_tr, Y_tr, X_te, Y_te] = train_test_split(X, Y, n_train, n_test);

for i = 1:length(n_unlabelled_range)
    
    %% split into labelled and unlabelled
    n_unlabelled = n_unlabelled_range(i);
    [X_lab, Y_lab, X_unl, ~] = train_test_split(X_tr, Y_tr,...
        n_labelled, n_unlabelled);
    switch dataset
        case 'diabetes'     
            X_lab_cau = X_lab(:, 1:end-1); X_lab_eff = X_lab(:, end);
            X_unl_cau = X_unl(:, 1:end-1); X_unl_eff = X_unl(:, end);
        case 'heartdisease'
    end   
    
    %% Baseline 1: logistic regression on labelled data only
    [B_base,~,~] = mnrfit(X_lab, Y_lab);
    [res.AUC_base(i, iter), ~, ~] = evaluate(B_base, X_te, Y_te);

    %% Semi-Generative Model
    fun = @(theta) nll_pooled(X_lab_cau,X_lab_eff,Y_lab,...
        X_unl_cau,X_unl_eff,theta,lam_semigen);
    [~,p_cau] = size(X_lab_cau);
    th_0 = zeros(3*p_cau+5,1);
%     th_0 = 0.1 * randn(3*p_cau+5,1);
%     theta = fminsearch(fun,th_0, options);
    
    %% Conditional label propagation
    [X_lab_cau, X_lab_eff,Y_lab, X_unl_cau, X_unl_eff] = lin_fun_label_prop(...
        X_lab_cau, X_lab_eff, Y_lab, X_unl_cau, X_unl_eff, lam_ridge, threshold);
    [B_clp,~,~] = mnrfit([X_lab_cau,X_lab_eff], Y_lab);
    [res.AUC_clp(i, iter), ~, ~] = evaluate(B_clp, X_te, Y_te);
end
toc
end

%%
AUCs = figure(2); hold on; xlabel('log_2(unlabelled examples)'); ylabel('AUROC');
errorbar(log2(n_unlabelled_range), mean(res.AUC_base,2), std(res.AUC_base'), 'LineWidth', 2);
errorbar(log2(n_unlabelled_range), mean(res.AUC_clp,2), std(res.AUC_clp'), 'LineWidth', 2);
legend('baseline sup. LR', 'lin. func. label prop + LR');
title(sprintf('%i labelled examples', n_labelled))




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
    
