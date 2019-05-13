close all; clear all; clc;
% load and preprocess data
T = readtable('heart.csv'); A = table2array(T);
X = A(:,1:end-1); Y = A(:,end)+1;
X = (X - mean(X))./std(X);

% analyse dataset
[B_full,~,stats] = mnrfit(X, Y);
[AUC_full, x_roc_full, y_roc_full] = evaluate(B_full, X, Y);

%%

feat = [1, 2, 6, 7]; %Pregnancy, Glucose, BMI, PedigreeFunc. (p < 0.01)
[B_feat,~,~] = mnrfit(X(:,feat), Y);
[AUC_feat, x_roc_feat, y_roc_feat] = evaluate(B_feat, X(:,feat), Y);

% visualize
model_comparison = figure(1); hold on;
plot(x_roc_full,y_roc_full); plot(x_roc_feat,y_roc_feat); plot([0 1],[0 1]);
hold off; xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('Comparison of full and reduced model')
legend(sprintf('full (AUC: %.3f)', round(AUC_full,4)),...
    sprintf('reduced (AUC: %.3f)', round(AUC_feat,4)));

glucose_plot = figure(2);
Preg = X(:,1); Gluc = X(:,2); BMI = X(:,6); PedFun = X(:,7); Age = X(:,8);
healthy = histogram(Gluc(Y==1), 20); hold on;
sick = histogram(Gluc(Y==2), 20);
xlabel('Plasma glucose in tolerance test')
ylabel('Count')
title('Effect feature distribution')
legend('healthy', 'diabetic')

cause_effect_plot = figure(3);
subplot(1,3,1)
    plot(BMI(Y==1), Gluc(Y==1), 'b.'); hold on;
    plot(BMI(Y==2), Gluc(Y==2), 'r.'); hold off;
    title('Glucose vs BMI'); legend('healthy', 'diabetic'); 
    xlabel('BMI (kg/m^2)'); ylabel('glucose')
subplot(1,3,2)
    plot(Preg(Y==1), Gluc(Y==1), 'b.'); hold on;
    plot(Preg(Y==2), Gluc(Y==2), 'r.'); hold off;
    title('Glucose vs pregnancies'); legend('healthy', 'diabetic'); 
    xlabel('Pregnancies'); ylabel('glucose')
subplot(1,3,3)
    plot(PedFun(Y==1), Gluc(Y==1), 'b.'); hold on;
    plot(PedFun(Y==2), Gluc(Y==2), 'r.'); hold off;
    title('Glucose vs diabetes pedigree function'); legend('healthy', 'diabetic'); 
    xlabel('diabetes pedigree function'); ylabel('glucose')
    
% ridge regression of Glucose on causal features
X_cau = [BMI, Preg, PedFun];
b_full = ridge(Gluc, X_cau, 1:1:10, 0);
b_healthy = ridge(Gluc(Y==1), X_cau(Y==1,:), 1:1:10, 0);
b_diabetic = ridge(Gluc(Y==2), X_cau(Y==2, :), 1:1:10, 0);

lm_full = fitlm(X_cau,Gluc)
lm_healthy = fitlm(X_cau(Y==1,:), Gluc(Y==1))
lm_diabetic = fitlm(X_cau(Y==2,:), Gluc(Y==2))
