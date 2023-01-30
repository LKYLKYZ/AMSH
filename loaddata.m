function [LTrain, LTest,XTrain,XTest,YTrain,YTest] = loaddata(dataset)
    load(dataset);
    LTrain = L_tr;
    LTest = L_te;
    XTrain = I_tr;
    YTrain = T_tr;
    XTest = I_te;
    YTest = T_te;
end
