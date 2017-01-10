function [trainX, testX] = tf_idf_tt(trainTF, testTF)
% TF-IDF weighting
% ([1+log(tf)]*log[N/df])
[n,m] = size(trainTF);  % the number of (training) documents and terms
df = sum(trainTF>0);  % (training) document frequency
d = sum(df>0); % the number of dimensions, i.e., terms occurred in (training) documents
[dfY, dfI] = sort(df, 2, 'descend');
trainTF = trainTF(:,dfI(1:d));
testTF = testTF(:,dfI(1:d));
idf = log(n./dfY(1:d));
IDF = sparse(1:d,1:d,idf);
[trainI,trainJ,trainV] = find(trainTF);
trainLogTF = sparse(trainI,trainJ,1+log(trainV),size(trainTF,1),size(trainTF,2));
[testI,testJ,testV] = find(testTF);
testLogTF = sparse(testI,testJ,1+log(testV),size(testTF,1),size(testTF,2));
trainX = trainLogTF*IDF;
testX = testLogTF*IDF;

end
