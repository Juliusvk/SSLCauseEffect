function [AUC, x_roc, y_roc] = evaluate(B,X,Y)
%   Detailed explanation goes here
pihat = mnrval(B,X);
[x_roc,y_roc,~,AUC] = perfcurve(Y, pihat(:,2), 2);

end

