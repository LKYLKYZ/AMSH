function [V] = solve_V(Z1,n,nbits)
%SOLVE_V 此处显示有关此函数的摘要
%   此处显示详细说明
     Z = Z1';
     Temp = Z'*Z-1/n*(Z'*ones(n,1)*(ones(1,n)*Z));
     [~,Lmd,QQ] = svd(Temp); clear Temp
     idx = (diag(Lmd)>1e-4);
     Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
     Pt = (Z-1/n*ones(n,1)*(ones(1,n)*Z)) *  (Q / (sqrt(Lmd(idx,idx))));
     P_ = orth(randn(n,nbits-length(find(idx==1))));
     V = sqrt(n)*[Pt P_]*[Q Q_]';
     V = V';
end

