function Y = LSA(X, nLowVec)

k = nLowVec;

[Y,~,~] = svds(X,k);

end
