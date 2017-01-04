function Xn = normalize(X)
% Normalize all feature vectors to unit length

n = size(X,1);  % the number of documents
Xt = X';
l = sqrt(sum(Xt.^2));  % the row vector length (L2 norm)
Ni = sparse(1:n,1:n,l);
Ni(Ni>0) = 1./Ni(Ni>0);
Xn = (Xt*Ni)';

end
