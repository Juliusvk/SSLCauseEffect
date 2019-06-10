%% load and preprocess data

close all; clear all; clc;
T = readtable('heart.csv'); A = table2array(T); 
% exclude 2 points for which thal is missing
X = A(A(:,13)~=0,1:end-1); Y = A(A(:,13)~=0,end)+1;

[B_full,~,stats_full] = mnrfit(X, Y);
[AUC_full, x_roc_full, y_roc_full] = evaluate(B_full, X, Y);

% Significant (p<0.05) causal features
ca = X(:,12); sex = X(:,2); trestbps = X(:,4);
% Significant (p<0.05) effect features
cp = X(:,3); thal = X(:,13); oldpeak = X(:,10); exang = X(:,9); thalach = X(:,8);

% Dummy code categorical variables
dummy_cp = dummyvar(cp+1); dcp = dummy_cp(:,1:end-1);
dummy_thal = dummyvar(thal); dthal = dummy_thal(:,1:end-1);

% Standardize numerical variables
trestbps = (trestbps - mean(trestbps))./std(trestbps);
thalach = (thalach - mean(thalach))./std(thalach);
oldpeak = (oldpeak - mean(oldpeak))./std(oldpeak);
ca = (ca - mean(ca))./std(ca);

X_C = [ca, sex, trestbps];
X_E = [dcp, dthal, oldpeak, exang, thalach];
X_min = [ca, sex, dcp, dthal];

% analysis of feature selection
[B_preproc,~,stats_preproc] = mnrfit([X_C, X_E], Y);
[AUC_preproc,x_roc_preproc,y_roc_preproc] = evaluate(B_preproc,[X_C,X_E],Y);
[B_min,~,stats_min] = mnrfit(X_min, Y);
[AUC_min, x_roc_min, y_roc_min] = evaluate(B_min, X_min, Y);

model_comparison = figure(1); hold on;
plot(x_roc_full,y_roc_full); plot(x_roc_preproc,y_roc_preproc);
plot(x_roc_min,y_roc_min); plot([0 1],[0 1]);
hold off; xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('Comparison of full and reduced model')
legend(sprintf('full (AUC: %.3f)', AUC_full),...
    sprintf('only signif. feat. (AUC: %.3f)', AUC_preproc), ...
    sprintf('only min. feat. (AUC: %.3f)', AUC_min));

glucose_plot = figure(2);
subplot(4,2,1)
    histogram(cp(Y==1), 20); hold on; histogram(cp(Y==2), 20);
    xlabel('cp'); ylabel('Count'); title('cp'); legend('healthy', 'sick')
subplot(4,2,2)
    histogram(thal(Y==1), 20); hold on; histogram(thal(Y==2), 20);
    xlabel('thal'); ylabel('Count'); title('thal'); legend('healthy', 'sick')
subplot(4,2,3)
    histogram(oldpeak(Y==1), 20); hold on; histogram(oldpeak(Y==2), 20);
    xlabel('oldpeak'); ylabel('Count'); title('oldpeak'); legend('healthy', 'sick')
subplot(4,2,4)
    histogram(exang(Y==1), 20); hold on; histogram(exang(Y==2), 20);
    xlabel('exang'); ylabel('Count'); title('exang'); legend('healthy', 'sick')
subplot(4,2,5)
    histogram(thalach(Y==1), 20); hold on; histogram(thalach(Y==2), 20);
    xlabel('thalach'); ylabel('Count'); title('thalach'); legend('healthy', 'sick')
subplot(4,2,6)
    histogram(ca(Y==1), 20); hold on; histogram(ca(Y==2), 20);
    xlabel('ca'); ylabel('Count'); title('ca'); legend('healthy', 'sick')
subplot(4,2,7)
    histogram(sex(Y==1), 20); hold on; histogram(sex(Y==2), 20);
    xlabel('sex'); ylabel('Count'); title('sex'); legend('healthy', 'sick')
subplot(4,2,8)
    histogram(trestbps(Y==1), 20); hold on; histogram(trestbps(Y==2), 20);
    xlabel('trestbps'); ylabel('Count'); title('trestbps'); legend('healthy', 'sick')


    
%% check for class differences in the mechanisms between cause and effect
[C_full,~,s_full] = mnrfit(X_C, cp+1);
[C_1,~,s_1] = mnrfit(X_C(Y==1,:), cp(Y==1)+1);
[C_2,~,s_2] = mnrfit(X_C(Y==2,:), cp(Y==2)+1);






