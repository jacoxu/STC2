function Y = LapEig(X, options, d)
% Laplacian Eigenmap
W = constructW(X,options);

D = diag(sum(W));
ii = find(diag(D)==0);
if size(ii)~=0
    for i=1:size(ii)
        D(ii(i),ii(i)) = 0.01;
    end
end
D = sparse(D);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L = D - W;

options = [];
options.disp = 0;
[eigVecs,eigVals] = eigs(L,D,1+d,'sa',options);
Y = eigVecs(:,2:end);

end
