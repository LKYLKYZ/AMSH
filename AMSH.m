function [B1,B2,B1_test,B2_test] = AMSH(X, Y, LX, LY, param, XTest, YTest)
    L1 = NormalizeFea(LX,0);
    L2 = NormalizeFea(LY,0);
    n1 = size(X,2);
    n2 = size(Y,2);

    beta = param.beta;
    lambda = param.lambda;
    eta = param.eta;
    r = param.nbits;
    nbits = param.nbits;


    sel_sample = X(:,randsample(n1, 1000),:);
    [pcaW, ~] = eigs(cov(sel_sample'), nbits);
    V1 = pcaW'*X;
    B1 = sign(V1);
    
    sel_sample = Y(:,randsample(n2, 1000),:);
    [pcaW, ~] = eigs(cov(sel_sample'), nbits);
    V2 = pcaW'*Y;
    B2 = sign(V2);

    R1 = ones(size(LX));
    R1(LX == 0) = -1;
    R2 = ones(size(LY));
    R2(LY == 0) = -1;
    
    E1 = zeros(size(R1));
    E2 = zeros(size(R2));

    for iter = 1:param.iter 

     P1 = (LX+R1.*E1)*V1'/(V1*V1');
     P2 = (LY+R2.*E2)*V2'/(V2*V2');

     Z = P1'*(LX+R1.*E1)+lambda*r*B1*L1'*L1+eta*B1+beta*r*V2*L2'*L1;
     V1 = solve_V(Z,n1,nbits);

     Z = P2'*(LY+R2.*E2)+lambda*r*B2*L2'*L2+eta*B2+beta*r*V1*L1'*L2;
     V2 = solve_V(Z,n2,nbits);

     B1 = sign(lambda*r*V1*L1'*L1 + eta*V1);
     B2 = sign(lambda*r*V2*L2'*L2 + eta*V2);

     E1 = max((P1*V1-LX).*R1,0);
     E2 = max((P2*V2-LY).*R2,0);
end
M1 = zeros(size(B1));
M2 = zeros(size(B2));

for iter = 1:param.iter 
    %update F
    T1 = B1 + B1.*M1;
    T2 = B2 + B2.*M2;

     F1 = T1*X'/(X*X'+ 0.01*eye(size(X,1)));
     F2 = T2*Y'/(Y*Y'+ 0.01*eye(size(Y,1)));


    K1 = F1*X-B1;
    K2 = F2*Y-B2;
    
    M1 = max(K1.*B1,0);
    M2 = max(K2.*B2,0);

end

B1 = double(B1 > 0)';
B2 = double(B2 > 0)';
B1_test = double(XTest*F1'>0);
B2_test = double(YTest*F2'>0);
end

