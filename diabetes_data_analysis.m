close all; clear all; clc;
% load and preprocess data
T = readtable('diabetes.csv'); A = table2array(T);
X = A(:,1:8); Y = A(:,9)+1;

%impute missing values by class mean instead of 0
X(X(:,2)==0,2) = (Y(X(:,2)==0)-1) * mean(X(and(Y==2,X(:,2)~=0), 2)) + ...
    (2-Y(X(:,2)==0)) * mean(X(and(Y==1, X(:,2)~=0), 2));
X(X(:,6)==0,6) = (Y(X(:,6)==0)-1) * mean(X(and(Y==2,X(:,6)~=0), 6)) + ...
    (2-Y(X(:,6)==0)) * mean(X(and(Y==1, X(:,6)~=0), 6));
X(X(:,3)==0,3) = (Y(X(:,3)==0)-1) * mean(X(and(Y==2,X(:,3)~=0), 3)) + ...
    (2-Y(X(:,3)==0)) * mean(X(and(Y==1, X(:,3)~=0), 3));
X(X(:,5)==0,5) = (Y(X(:,5)==0)-1) * mean(X(and(Y==2,X(:,5)~=0), 5)) + ...
    (2-Y(X(:,5)==0)) * mean(X(and(Y==1, X(:,5)~=0), 5));

X = (X - mean(X))./std(X);

% analyse dataset
[B_full,~,~] = mnrfit(X, Y);
[AUC_full, x_roc_full, y_roc_full] = evaluate(B_full, X, Y);

feat = [1, 6, 7, 8, 2]; %Pregnancy, BMI, PedigreeFunc., Glucose (p < 0.01)
[B_feat,~,~] = mnrfit(X(:,feat), Y);
[AUC_feat, x_roc_feat, y_roc_feat] = evaluate(B_feat, X(:,feat), Y);


% visualize
model_comparison = figure(1); hold on;
plot(x_roc_full,y_roc_full); plot(x_roc_feat,y_roc_feat); plot([0 1],[0 1]);
hold off; xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('Comparison of full and reduced model')
legend(sprintf('full (AUC: %.3f)', round(AUC_full,4)),...
    sprintf('reduced (AUC: %.3f)', round(AUC_feat,4)));


Preg = X(:,1); Gluc = X(:,2); BMI = X(:,6); PedFun = X(:,7); Age = X(:,8);
Insulin = X(:,5); Bloodpressure = X(:,3);

%effect_dist_plot = figure(2);

    
    
cause_effect= figure(3);
subplot(4,3,1)
    plot(BMI(Y==1), Gluc(Y==1), 'b.'); hold on;
    plot(BMI(Y==2), Gluc(Y==2), 'r.'); hold off;
    title('Glucose vs BMI'); legend('healthy', 'diabetic'); 
    xlabel('BMI (kg/m^2)'); ylabel('glucose')
subplot(4,3,2)
    plot(Preg(Y==1), Gluc(Y==1), 'b.'); hold on;
    plot(Preg(Y==2), Gluc(Y==2), 'r.'); hold off;
    title('Glucose vs pregnancies'); legend('healthy', 'diabetic'); 
    xlabel('Pregnancies'); ylabel('glucose')
subplot(4,3,3)
    plot(PedFun(Y==1), Gluc(Y==1), 'b.'); hold on;
    plot(PedFun(Y==2), Gluc(Y==2), 'r.'); hold off;
    title('Glucose vs pedigree function'); legend('healthy', 'diabetic'); 
    xlabel('Pedigree function'); ylabel('glucose')   
%cause_effect_insulin = figure(4);
subplot(4,3,4)
    plot(BMI(Y==1), Insulin(Y==1), 'b.'); hold on;
    plot(BMI(Y==2), Insulin(Y==2), 'r.'); hold off;
    title('Insulin vs BMI'); legend('healthy', 'diabetic'); 
    xlabel('BMI (kg/m^2)'); ylabel('Insulin')
subplot(4,3,5)
    plot(Preg(Y==1), Insulin(Y==1), 'b.'); hold on;
    plot(Preg(Y==2), Insulin(Y==2), 'r.'); hold off;
    title('Insulin vs pregnancies'); legend('healthy', 'diabetic'); 
    xlabel('Pregnancies'); ylabel('Insulin')
subplot(4,3,6)
    plot(PedFun(Y==1), Insulin(Y==1), 'b.'); hold on;
    plot(PedFun(Y==2), Insulin(Y==2), 'r.'); hold off;
    title('Insulin vs pedigree function'); legend('healthy', 'diabetic'); 
    xlabel('Pedigree function'); ylabel('Insulin'); 
%cause_effect_bloodpressure = figure(5);
subplot(4,3,7)
    plot(BMI(Y==1), Bloodpressure(Y==1), 'b.'); hold on;
    plot(BMI(Y==2), Bloodpressure(Y==2), 'r.'); hold off;
    title('Bloodpressure vs BMI'); legend('healthy', 'diabetic'); 
    xlabel('BMI (kg/m^2)'); ylabel('Bloodpressure')
subplot(4,3,8)
    plot(Preg(Y==1), Bloodpressure(Y==1), 'b.'); hold on;
    plot(Preg(Y==2), Bloodpressure(Y==2), 'r.'); hold off;
    title('Bloodpressure vs pregnancies'); legend('healthy', 'diabetic'); 
    xlabel('Pregnancies'); ylabel('Bloodpressure')
subplot(4,3,9)
    plot(PedFun(Y==1), Bloodpressure(Y==1), 'b.'); hold on;
    plot(PedFun(Y==2), Bloodpressure(Y==2), 'r.'); hold off;
    title('Bloodpressure vs pedigree function'); 
    legend('healthy', 'diabetic'); xlabel('Pedigree function');
    ylabel('Bloodpressure')
subplot(4,3,10)
    histogram(Gluc(Y==1), 20); hold on; histogram(Gluc(Y==2), 20);
    xlabel('Glucose'); ylabel('Count')
    title('Glucose'); legend('healthy', 'diabetic')
subplot(4,3,11)
    histogram(Insulin(Y==1), 20); hold on; histogram(Insulin(Y==2), 20);
    xlabel('Insulin'); ylabel('Count');
    title('Insulin'); legend('healthy', 'diabetic')
subplot(4,3,12)
    histogram(Bloodpressure(Y==1), 20); hold on;
    histogram(Bloodpressure(Y==2), 20); xlabel('Bloodpressure'); ylabel('Count');
    title('Bloodpressure'); legend('healthy', 'diabetic')
    
% ridge regression of Glucose on causal features
X_cau = [BMI, Preg, PedFun];
b_full = ridge(Gluc, X_cau, 1:1:10, 0);
b_healthy = ridge(Gluc(Y==1), X_cau(Y==1,:), 1:1:10, 0);
b_diabetic = ridge(Gluc(Y==2), X_cau(Y==2, :), 1:1:10, 0);

lm_full_gluc = fitlm(X_cau,Gluc)
lm_healthy_gluc = fitlm(X_cau(Y==1,:), Gluc(Y==1))
lm_diabetic_gluc = fitlm(X_cau(Y==2,:), Gluc(Y==2))

lm_full_insulin = fitlm(X_cau,Insulin)
lm_healthy_insulin = fitlm(X_cau(Y==1,:), Insulin(Y==1))
lm_diabetic_insulin = fitlm(X_cau(Y==2,:), Insulin(Y==2))

lm_full_bp = fitlm(X_cau,Bloodpressure)
lm_healthy_bp = fitlm(X_cau(Y==1,:), Bloodpressure(Y==1))
lm_diabetic_bp = fitlm(X_cau(Y==2,:), Bloodpressure(Y==2))



