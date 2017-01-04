function Train(dataset,p,filePrefixStr,evaluteScore,randinit4Methods,randinit4KMeans,repeatNum,nbcluster,parameters)
data_f = ['./../dataset/',dataset,'-STC2.mat'];
addpath(genpath('./autoDif'))
load(data_f,'all','all_lbl','fea_All','index','labels_All','size_vocab','sent_length','vocab_emb_Word2vec_48','vocab_emb_Word2vec_48_index');
%all-n*d data
Z = repmat(median(parameters.Y),size(parameters.Y,1),1);
all_lbl{1}=double(parameters.Y>Z);
%% ASSIGN P AND indices
vec1=all_lbl{1}(1,:);
p(9) = length(vec1); disp(strcat('Number of output classes:',num2str(p(9))));
p(2) = sent_length; disp(strcat('Max sent length:',num2str(p(2))));

p(30) = size_vocab; disp(strcat('Size vocab (and pad):',num2str(p(30)))); 

p(51) = 3;          disp(strcat('Local Connections Span at First layer:',num2str(p(51))));
p(52) = 3;          disp(strcat('Local Connections Span at Second layer:',num2str(p(52))));
p(53) = 3;          disp(strcat('Local Connections Span at Third layer:',num2str(p(53))));
%
disp(' ');
p(20) = 1e-4;       disp(strcat('Reg E (word vectors):',num2str(p(20))));
p(21) = 3e-5;       disp(strcat('Reg 1 (first conv layer):',num2str(p(21))));
p(22) = 3e-6;       disp(strcat('Reg 2 (second conv layer):',num2str(p(22))));
p(23) = 1e-5;       disp(strcat('Reg 3 (third conv layer):',num2str(p(23))));
p(24) = 1e-4;       disp(strcat('Reg Z (classification layer):',num2str(p(24))));


p(60) = 0;          disp(strcat('Multiple logistic:',num2str(~p(60))));

p(73) = length(unique(labels_All));

[all_msk, p] = Masks(all, all_lbl, p);
topACC=0;

   CR = RCTM(p);
    if p(34) %if use external vocabulary
         vocab_emb_length = length(vocab_emb_Word2vec_48(1,:));
         if vocab_emb_length > size_vocab
             error(['Error, and the size fo vocab_emb is:',vocab_emb_length])
         end
         CR.E(1:p(1),vocab_emb_Word2vec_48_index) = vocab_emb_Word2vec_48(1:p(1),1:vocab_emb_length);
    end
    CR.E(:,p(30)) = zeros(size(CR.E,1),1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%% REPRESENTATION SIZES
    disp(' ')
    if(p(68))
        disp(['Layer 1: ' num2str(p(3)) ' maps of depth ' num2str((p(1)+p(69))/(p(12)*1+1))])
        if p(10) >= 2, disp(['Layer 2: ' num2str(p(5)) ' maps of depth ' num2str((p(1)+p(69))/((p(12)+1)*(p(13)+1)))]), end
        if p(10) == 3, disp(['Layer 3: ' num2str(p(37)) ' maps of depth ' num2str((p(1)+p(69))/((p(12)+1)*(p(13)+1)*(p(35)+1)))]), end
    else
        disp(['Layer 1: ' num2str(p(3)) ' maps of depth ' num2str(p(1)/(p(12)*1+1))])
        if p(10) >= 2, disp(['Layer 2: ' num2str(p(5)) ' maps of depth ' num2str(p(1)/((p(12)+1)*(p(13)+1)))]), end
        if p(10) == 3, disp(['Layer 3: ' num2str(p(37)) ' maps of depth ' num2str(p(1)/((p(12)+1)*(p(13)+1)*(p(35)+1)))]), end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    all_lbl = all_lbl{1}; %getting rid of length information for sentences


    %% TRAINING
    [X, decodeInfo] = param2stack(CR.E, CR.one, CR.one_b, CR.two, CR.two_b, CR.three, CR.three_b, CR.Z, p);
    disp(strcat('Total number of parameters: ', num2str(numel(X))));
    disp(' ');

    gamma = p(64); %Learning rate
    batchsize = p(61);
    maxEpochs = p(62);%199;

    disp(strcat('Learning rate:',num2str(gamma)));
    disp(strcat('Batch size:',num2str(batchsize)));
    disp(strcat('Max Number of epochs:',num2str(maxEpochs)));

    num_batch_epochs = floor(size(all,1)/(batchsize)); %leaves last batch out at an iteration
    indices = kron(1:p(1),ones(1,batchsize*p(2)+1)).';

    currentEpoch=0;
    AC = 1000;
    breakEpoch = 0;
    for i=1:maxEpochs 
        rand('state',randinit4Methods)
        randn('state',randinit4Methods)
        disp(['state randinit4Methods:',num2str(randinit4Methods)]);

        permut = randperm(size(all,1)); 
        all= all(permut,:); 
        all_lbl = all_lbl(permut,:); 

       %%
        all_msk = all_msk(permut,:);
        gradient_hist = zeros(length(X),1); %TODO NOTE: resetting could be done less often. Parametrize this
        %gradient_hist
        disp(['Epoch:' num2str(i) '/' num2str(maxEpochs)]);

        if i>p(33)
           p(32) = 0; %turn off embedding learning after a few epochs if p(32)==1
        end

        for j=1:num_batch_epochs
            minibatch = reshape(all((j-1)*batchsize+1:j*batchsize,:)',1,[]);%fixed size batches 
            labels = all_lbl((j-1)*batchsize+1:j*batchsize,:); 
            mini_msk = all_msk((j-1)*batchsize+1:j*batchsize,:); 

            if 0, fastDerivativeCheck(@CostFunction,X,1,2, decodeInfo, minibatch, labels, mini_msk, indices, p); end
           
            [cost,grad]=CostFunction(X, decodeInfo, minibatch, labels, mini_msk, indices, p);

            if j <= 10000 %Only print PPL at the beginning 
                disp(['J:' num2str(j) ' PPL-num2str(cost): ' num2str(cost)]);
            end

            gradient_hist = gradient_hist + grad.^2; 
            sq = sqrt(gradient_hist);
            sq(sq~=0) = gamma./sq(sq~=0);

            X = X-sq.*grad;
            
        end
        disp(['Average Parameter Weight: ', num2str(sum(abs(X))/length(X))]);
    end
    disp('End iterator ...!');
    [tmpACC,tmpNMI,evaluteScore] = evaluateCluster(data_f, X,decodeInfo,p,randinit4KMeans,evaluteScore,i,repeatNum,nbcluster);
    disp(['Current ACC:',num2str(tmpACC),' and NMI:',num2str(tmpNMI)]);
    save(['./results/evaluteScore_',parameters.method,'_results.mat'], 'evaluteScore');

end

function [tmpACC,tmpNMI,evaluteScore] = evaluateCluster(data_f, X,decodeInfo,p,randinit4KMeans,evaluteScore,index,repeatNum,nbcluster)
    load(data_f,'all', 'all_lbl', 'labels_All');
    [all_msk, p] = Masks(all, all_lbl, p);
    % reshape to one-dim vector
    all_batch = reshape(all',1,[]);
    
    disp('Start predict the all data for cluster!');
    batch_size = 200;
    all_semantic = [];
    for b=1:ceil(size(all,1)/batch_size) 
        all_miniBatch = all_batch(((b-1)*batch_size*p(2))+1:min(b*batch_size*p(2),end));
        all_miniMsk = all_msk((b-1)*batch_size+1:min(b*batch_size,end),:);
        tmp_semantic = Predict(X, decodeInfo, all_miniBatch, all_miniMsk, p);
        all_semantic = [all_semantic;tmp_semantic'];
    end
%     save(['./results/all_semantic_',parameters.method,'_results.mat'], 'all_semantic');
    disp('Start evluate the data for clustering!');
    index_sub = 1;
    for i=1:repeatNum
        rand('state',randinit4KMeans) 
        randn('state',randinit4KMeans) 
        disp(['state randinit4KMeans:',num2str(randinit4KMeans)]);
        randinit4KMeans = randinit4KMeans+1;
        res = kmeans(all_semantic, nbcluster);
        res = bestMap(labels_All,res);
        %=============  evaluate AC: accuracy ========================================
        AC = length(find(labels_All == res))/length(labels_All)*100;
        %=============  evaluate MIhat: nomalized mutual information =================
        MIhat = MutualInfo(labels_All,res)*100;
        evaluteScore(index_sub,(index-1)*2+1)=AC;
        evaluteScore(index_sub,(index-1)*2+2)=MIhat;
        index_sub=index_sub+1;
    end
    tmpACC = mean(evaluteScore(:,(index-1)*2+1));
    tmpNMI = mean(evaluteScore(:,(index-1)*2+2));
end