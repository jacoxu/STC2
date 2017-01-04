%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Self-Taught Convolutional Neural Networks for Short Text Clustering
% Accepted for publication in Neural Networks -- 2016/12/27
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;clear;
addpath(genpath([pwd '/']));
%% Please select one clustering method here.
% Kmeans, RecNN, AveEmbedding, LSA, Spectral_LE, LPI,STC2_AE, STC2_LSA, STC2_LE, STC2_LPI,
% para2vecs, skip_uni, skip_bi, skip_combine
% NOTE: for para2vecs and skip_thought methods, should first generate the text vectors
method='Kmeans'; 
parameters.weightMode=1; % 0 - TF or 1 - TFIDF weighting for Kmeans and AveEmbedding methods.
dataset='Biomedical'; % SearchSnippets, StackOverflow or Biomedical
index=0;
randinit4Methods=0;
randinit4KMeans=0;
repeatNum=10;
evaluteScore=[];
parameters.wordDim=48;
parameters.nLowVecFixed=1; % defalut is 1, nonFiexed for LSA, Spectral_LE, LPI to brute search the best vec from 10:10:200
%%
if (strcmp(dataset,'SearchSnippets'));    nLowVec =8;
elseif (strcmp(dataset,'StackOverflow')); nLowVec =20;
elseif (strcmp(dataset,'Biomedical'));    nLowVec =20;
else error(['Please input a right dataset name rather than ''',dataset,''''])
end
evaluteScore=STC2(method,nLowVec,randinit4Methods,randinit4KMeans,evaluteScore,dataset,index,repeatNum,parameters);

if((~parameters.nLowVecFixed)&&(~(strncmpi('STC2',method,4)))) % just for LSA, Spectral_LE, LPI
    for nLowVec=10:10:200
        index=index+1;
        evaluteScore=STC2(method,nLowVec,randinit4Methods,randinit4KMeans,evaluteScore,dataset,index,repeatNum,parameters);
    end
end

if (~(strncmpi('STC2',method,4)))
    save(['./results/evaluteScore_',method,'_results.mat'], 'evaluteScore');
end