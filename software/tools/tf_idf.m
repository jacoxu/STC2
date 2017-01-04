function [trainX] = tf_idf(trainTF)
% TF-IDF weighting
% ([1+log(tf)]*log[N/df])
[n,m] = size(trainTF);  % the number of (training) documents and terms
df = sum(trainTF>0);  % (training) document frequency
idf = log(n./df);
IDF = sparse(1:m,1:m,idf);
[trainI,trainJ,trainV] = find(trainTF);
trainLogTF = sparse(trainI,trainJ,1+log(trainV),size(trainTF,1),size(trainTF,2));
trainX = trainLogTF*IDF;
end
