close all; clc; clear all;

%% load and preprocess data
T = readtable('diabetes.csv'); A = table2array(T);
X = A(:,1:8); Y = A(:,9)+1;

% train and test on full dataset: oracle
[B_full,~,~] = mnrfit(X, Y);
[AUC_full, x_roc_full, y_roc_full] = evaluate(B_full, X, Y);

% Feature selection: Pregnancy, Glucose, BMI, PedigreeFunc. (p < 0.01)
X = A(:, [1, 2, 6, 7]); 

% model_comparison = figure(1); hold on;
% xlabel('False Positive Rate'); ylabel('True Positive Rate');
% plot([0 1],[0 1], 'DisplayName', 'random','LineWidth', 2);

n_train = 512; n_test = 256; n_iter = 100; n_labelled_range = 2.^(4:9);
res.AUC = zeros(length(n_labelled_range), n_iter);

for iter = 1:n_iter
%% separate into train and test data
[X_tr, Y_tr, X_te, Y_te] = train_test_split(X, Y, n_train, n_test);

for i = 1:length(n_labelled_range)
    %% split into labelled and unlabelled
    n_labelled = n_labelled_range(i);
    [X_lab, Y_lab, X_unl, Y_unl] = train_test_split(X_tr, Y_tr,...
        n_labelled, n_train - n_labelled);

    %% Baseline1: logistic regression on labelled data only
    [B,~,~] = mnrfit(X_lab, Y_lab);
    [res.AUC(i, iter), x_roc, y_roc] = evaluate(B, X_te, Y_te);
%     txt = sprintf('%i labelled, AUC = %.3f', n_labelled, res.AUC(i));
%     plot(x_roc,y_roc, 'DisplayName', txt, 'LineWidth', 2);

end
end
% hold off; legend show;

%%
AUCs = figure(2); plot(log2(n_labelled_range), mean(res.AUC,2));
xlabel('labelled examples'); ylabel('AUC');
