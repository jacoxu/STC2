function evaluteScore=STC2(method,nLowVec, randinit4Methods,randinit4KMeans,evaluteScore,dataset,index,repeatNum,parameters)

dataStr=['./../dataset/',dataset,'-lite.mat'];
load(dataStr);
index_sub = 1;
parameters.method = method;
%% init parameters
options = [];
options.NeighborMode = 'KNN';
options.Metric = 'Euclidean';
options.WeightMode = 'HeatKernel';
options.t = 1;
options.bSelfConnected = 0;
options.k = 15;
%% set the clustering number
if (strcmp(dataset,'SearchSnippets'));    nbcluster = 8;
elseif (strcmp(dataset,'StackOverflow')); nbcluster = 20;
elseif (strcmp(dataset,'Biomedical'));    nbcluster = 20;    
end

%%
if (strcmp(method,'Kmeans'))
    warning(['K-means clustering is very slow on original high-dimensionality (',num2str(length(fea(1,:))),' dim.) text features.'])
    disp('If you want to run clustering via KMeans, please press any key to continue and have a little patience...')
    disp('If you want to stop this program now, please press Ctrl + c.')
    pause;
    disp('Here is the baseline clustering method: Kmeans!'); 
    if parameters.weightMode
        fea=tf_idf(fea);
    end
    fea = normalize(fea);
    for i=1:repeatNum
        rand('state',randinit4KMeans)
        randn('state',randinit4KMeans)
        disp(['state randinit4KMeans:',num2str(randinit4KMeans)]);
        randinit4KMeans = randinit4KMeans+1;
        res = kmeans(fea, nbcluster);
        res = bestMap(gnd,res);
        %=============  evaluate AC: accuracy ========================================
        AC = length(find(gnd == res))/length(gnd)*100;
        %=============  evaluate MIhat: nomalized mutual information =================
        MIhat = MutualInfo(gnd,res)*100;
        disp(['Evalute:',num2str(i),', ACC is ',num2str(AC),' and NMI is ',num2str(MIhat)]);
        evaluteScore(index_sub,index*2+1)=AC;
        evaluteScore(index_sub,index*2+2)=MIhat;
        index_sub=index_sub+1;
    end
    disp(['Final evalute results: mean ACC is ',num2str(mean(evaluteScore(:,index*2+1))),...
                            ' and mean NMI is ',num2str(mean(evaluteScore(:,index*2+2)))]);  
elseif (strcmp(method,'RecNN'))
    disp('Here is the baseline clustering method: RecNN!');       
    rand('state',randinit4Methods)
    randn('state',randinit4Methods)
    disp(['state randinit4Methods:',num2str(randinit4Methods)]);
    Y = RecNN(randinit4Methods,randinit4KMeans,dataset,parameters);
    disp('Has obtained the learned features from recursive NN ..., and start KMeans ...');
    disp('Start KMeans to cluster top features...');
    fea = normalize(Y(:,:,1));
    index_sub=1;
    for i=1:repeatNum
        rand('state',randinit4KMeans)
        randn('state',randinit4KMeans)
        disp(['state randinit4KMeans:',num2str(randinit4KMeans)]);
        randinit4KMeans = randinit4KMeans+1;
        res = kmeans(fea, nbcluster);
        res = bestMap(gnd,res);
        %=============  evaluate AC: accuracy ========================================
        AC = length(find(gnd == res))/length(gnd)*100;
        %=============  evaluate MIhat: nomalized mutual information =================
        MIhat = MutualInfo(gnd,res)*100;
        disp(['Evalute:',num2str(i),', ACC is ',num2str(AC),' and NMI is ',num2str(MIhat)]);
        evaluteScore(index_sub,index*2+1)=AC;
        evaluteScore(index_sub,index*2+2)=MIhat;
        index_sub=index_sub+1;
    end
    disp(['Final evalute results: mean ACC is ',num2str(mean(evaluteScore(:,index*2+1))),...
                            ' and mean NMI is ',num2str(mean(evaluteScore(:,index*2+2)))]);  
    index=index+1;
    randinit4KMeans=0;
    fea = normalize(Y(:,:,2));
    disp('Start KMeans to cluster average features...');
    index_sub=1;
    for i=1:repeatNum
        rand('state',randinit4KMeans)
        randn('state',randinit4KMeans)
        disp(['state randinit4KMeans:',num2str(randinit4KMeans)]);
        randinit4KMeans = randinit4KMeans+1;
        res = kmeans(fea, nbcluster);
        res = bestMap(gnd,res);
        %=============  evaluate AC: accuracy ========================================
        AC = length(find(gnd == res))/length(gnd)*100;
        %=============  evaluate MIhat: nomalized mutual information =================
        MIhat = MutualInfo(gnd,res)*100;
        disp(['Evalute:',num2str(i),', ACC is ',num2str(AC),' and NMI is ',num2str(MIhat)]);
        evaluteScore(index_sub,index*2+1)=AC;
        evaluteScore(index_sub,index*2+2)=MIhat;
        index_sub=index_sub+1;
    end
    disp(['Final evalute results: mean ACC is ',num2str(mean(evaluteScore(:,index*2+1))),...
                            ' and mean NMI is ',num2str(mean(evaluteScore(:,index*2+2)))]);      
    index=index+1;
    randinit4KMeans=0;
    [t1, t2, t3] = size(Y);
    Y = reshape(Y,t1, t2*t3);    
    fea = normalize(Y);
    disp('Start KMeans to cluster top plus average features...');
    index_sub=1;
    for i=1:repeatNum
        rand('state',randinit4KMeans)
        randn('state',randinit4KMeans)
        disp(['state randinit4KMeans:',num2str(randinit4KMeans)]);
        randinit4KMeans = randinit4KMeans+1;
        res = kmeans(fea, nbcluster);
        res = bestMap(gnd,res);
        %=============  evaluate AC: accuracy ========================================
        AC = length(find(gnd == res))/length(gnd)*100;
        %=============  evaluate MIhat: nomalized mutual information =================
        MIhat = MutualInfo(gnd,res)*100;
        disp(['Evalute:',num2str(i),', ACC is ',num2str(AC),' and NMI is ',num2str(MIhat)]);
        evaluteScore(index_sub,index*2+1)=AC;
        evaluteScore(index_sub,index*2+2)=MIhat;
        index_sub=index_sub+1;
    end
    disp(['Final evalute results: mean ACC is ',num2str(mean(evaluteScore(:,index*2+1))),...
                            ' and mean NMI is ',num2str(mean(evaluteScore(:,index*2+2)))]);      
elseif (strcmp(method,'AveEmbedding'))
    disp('Here is baseline clustering method: AveEmbedding!');    
    rand('state',randinit4Methods) 
    randn('state',randinit4Methods) 
    disp(['state randinit4Methods:',num2str(randinit4Methods)]);
    Y = AveEmbedding(dataset,parameters);
    for i=1:repeatNum
        rand('state',randinit4KMeans)
        randn('state',randinit4KMeans)
        disp(['state randinit4KMeans:',num2str(randinit4KMeans)]);
        randinit4KMeans = randinit4KMeans+1;
        res = kmeans(Y, nbcluster);
        res = bestMap(gnd,res);
        %=============  evaluate AC: accuracy ========================================
        AC = length(find(gnd == res))/length(gnd)*100;
        %=============  evaluate MIhat: nomalized mutual information =================
        MIhat = MutualInfo(gnd,res)*100;
        disp(['Evalute:',num2str(i),', ACC is ',num2str(AC),' and NMI is ',num2str(MIhat)]);
        evaluteScore(index_sub,index*2+1)=AC;
        evaluteScore(index_sub,index*2+2)=MIhat;
        index_sub=index_sub+1;
    end
    disp(['Final evalute results: mean ACC is ',num2str(mean(evaluteScore(:,index*2+1))),...
                            ' and mean NMI is ',num2str(mean(evaluteScore(:,index*2+2)))]);   
elseif (strcmp(method,'LSA'))
    disp('Here is baseline clustering method: LSA!');  
    if (parameters.nLowVecFixed)
       if (strcmp(dataset,'SearchSnippets'));    nLowVec= 10;
       elseif (strcmp(dataset,'StackOverflow')); nLowVec= 20;
       elseif (strcmp(dataset,'Biomedical'));    nLowVec= 20;
       end 
    end
    fea=tf_idf(fea);
    fea = normalize(fea);
    rand('state',randinit4Methods)
    randn('state',randinit4Methods)
    disp(['state randinit4Methods:',num2str(randinit4Methods)]);
    disp(['nLowVec is:',num2str(nLowVec)]);
    Y = LSA(fea,nLowVec);
    Y = normalize(Y);    
    for i=1:repeatNum
        rand('state',randinit4KMeans)
        randn('state',randinit4KMeans)
        disp(['state randinit4KMeans:',num2str(randinit4KMeans)]);
        randinit4KMeans = randinit4KMeans+1;
        res = kmeans(Y, nbcluster);
        res = bestMap(gnd,res);
        %=============  evaluate AC: accuracy ========================================
        AC = length(find(gnd == res))/length(gnd)*100;
        %=============  evaluate MIhat: nomalized mutual information =================
        MIhat = MutualInfo(gnd,res)*100;
        disp(['Evalute:',num2str(i),', ACC is ',num2str(AC),' and NMI is ',num2str(MIhat)]);
        evaluteScore(index_sub,index*2+1)=AC;
        evaluteScore(index_sub,index*2+2)=MIhat;
        index_sub=index_sub+1;
    end
    disp(['Final evalute results: mean ACC is ',num2str(mean(evaluteScore(:,index*2+1))),...
                            ' and mean NMI is ',num2str(mean(evaluteScore(:,index*2+2)))]);                            
elseif (strcmp(method,'Spectral_LE'))
    disp('Here is baseline clustering method: Spectral_LE!');
    if (parameters.nLowVecFixed)
       if (strcmp(dataset,'SearchSnippets'));    nLowVec= 20;
       elseif (strcmp(dataset,'StackOverflow')); nLowVec= 70;
       elseif (strcmp(dataset,'Biomedical'));    nLowVec= 30;
       end
    end
    fea=tf_idf(fea);
    fea = normalize(fea);
    rand('state',randinit4Methods)
    randn('state',randinit4Methods)
    disp(['state randinit4Methods:',num2str(randinit4Methods)]);
    disp(['nLowVec is:',num2str(nLowVec)]);
    Y = LapEig(fea,options,nLowVec);
    Y = normalize(Y);
    for i=1:repeatNum
        rand('state',randinit4KMeans)
        randn('state',randinit4KMeans)
        disp(['state randinit4KMeans:',num2str(randinit4KMeans)]);
        randinit4KMeans = randinit4KMeans+1;
        res = kmeans(Y, nbcluster);
        res = bestMap(gnd,res);
        %=============  evaluate AC: accuracy ========================================
        AC = length(find(gnd == res))/length(gnd)*100;
        %=============  evaluate MIhat: nomalized mutual information =================
        MIhat = MutualInfo(gnd,res)*100;
        disp(['Evalute:',num2str(i),', ACC is ',num2str(AC),' and NMI is ',num2str(MIhat)]);
        evaluteScore(index_sub,index*2+1)=AC;
        evaluteScore(index_sub,index*2+2)=MIhat;
        index_sub=index_sub+1;
    end
    disp(['Final evalute results: mean ACC is ',num2str(mean(evaluteScore(:,index*2+1))),...
                            ' and mean NMI is ',num2str(mean(evaluteScore(:,index*2+2)))]);      
elseif (strcmp(method,'LPI'))
    disp('Here is baseline clustering method: LPI!');
    if (parameters.nLowVecFixed)
       if (strcmp(dataset,'SearchSnippets'));    nLowVec= 20;
       elseif (strcmp(dataset,'StackOverflow')); nLowVec= 80;
       elseif (strcmp(dataset,'Biomedical'));    nLowVec= 30;
       end
    end
    fea=tf_idf(fea);
    fea = normalize(fea);
    rand('state',randinit4Methods)
    randn('state',randinit4Methods)
    disp(['state randinit4Methods:',num2str(randinit4Methods)]);
    W = constructW(fea,options);
    options.PCARatio = 1;
    options.ReducedDim=nLowVec;
    [eigvector, ~] = lpp(W, options, fea);
    disp(['nLowVec is:',num2str(nLowVec)]);
    Y = fea*eigvector;
    Y = normalize(Y);    
    for i=1:repeatNum
        rand('state',randinit4KMeans)
        randn('state',randinit4KMeans)
        disp(['state randinit4KMeans:',num2str(randinit4KMeans)]);
        randinit4KMeans = randinit4KMeans+1;
        res = kmeans(Y, nbcluster);
        res = bestMap(gnd,res);
        %=============  evaluate AC: accuracy ========================================
        AC = length(find(gnd == res))/length(gnd)*100;
        %=============  evaluate MIhat: nomalized mutual information =================
        MIhat = MutualInfo(gnd,res)*100;
        disp(['Evalute:',num2str(i),', ACC is ',num2str(AC),' and NMI is ',num2str(MIhat)]);
        evaluteScore(index_sub,index*2+1)=AC;
        evaluteScore(index_sub,index*2+2)=MIhat;
        index_sub=index_sub+1;
    end    
    disp(['Final evalute results: mean ACC is ',num2str(mean(evaluteScore(:,index*2+1))),...
                            ' and mean NMI is ',num2str(mean(evaluteScore(:,index*2+2)))]);     
elseif (strcmp(method,'STC2_AE'))
    disp('Here is the proposed method: STC2_AE!'); 
    rand('state',randinit4Methods)
    randn('state',randinit4Methods)
    disp(['state randinit4Methods:',num2str(randinit4Methods),' and start to binary embedding!']);
    disp('The Embedding is static, so we initialize the word Embedding here!')
    parameters.weightMode=0;
    Y = AveEmbedding(dataset,parameters);
    parameters.Y = Y;
    if (strcmp(dataset,'SearchSnippets'));    parameters.iter = 40;
    elseif (strcmp(dataset,'StackOverflow')); parameters.iter = 10;
    elseif (strcmp(dataset,'Biomedical'));    parameters.iter = 4;
    end
    parameters.iter = 10;
    STC2_CNN(dataset,parameters,evaluteScore,randinit4Methods,randinit4KMeans,repeatNum,nbcluster);                        
elseif (strcmp(method,'STC2_LSA'))
    disp('Here is the proposed method: STC2_LSA!');
    fea=tf_idf(fea);
    fea = normalize(fea);    
    rand('state',randinit4Methods) 
    randn('state',randinit4Methods) 
    disp(['state randinit4Methods:',num2str(randinit4Methods)]);
    if (strcmp(dataset,'SearchSnippets'));    nLowVec= 10;
    elseif (strcmp(dataset,'StackOverflow')); nLowVec= 20;
    elseif (strcmp(dataset,'Biomedical'));    nLowVec= 20;
    end 
    disp(['nLowVec is:',num2str(nLowVec)]);
    Y = LSA(fea,nLowVec);
    parameters.Y = Y;
    if (strcmp(dataset,'SearchSnippets'));    parameters.iter = 10;
    elseif (strcmp(dataset,'StackOverflow')); parameters.iter = 2;
    elseif (strcmp(dataset,'Biomedical'));    parameters.iter = 5;
    end
    STC2_CNN(dataset,parameters,evaluteScore,randinit4Methods,randinit4KMeans,repeatNum,nbcluster);
elseif (strcmp(method,'STC2_LE'))
    disp('Here is the proposed method: STC2_LE!');
    fea=tf_idf(fea);
    fea = normalize(fea);    
    rand('state',randinit4Methods) 
    randn('state',randinit4Methods) 
    disp(['state randinit4Methods:',num2str(randinit4Methods)]);
    if (strcmp(dataset,'SearchSnippets'));    nLowVec= 20;
    elseif (strcmp(dataset,'StackOverflow')); nLowVec= 70;
    elseif (strcmp(dataset,'Biomedical'));    nLowVec= 30;
    end 
    disp(['nLowVec is:',num2str(nLowVec)]);
    Y = LapEig(fea,options,nLowVec); 
    parameters.Y = Y;
    if (strcmp(dataset,'SearchSnippets'));    parameters.iter = 15;
    elseif (strcmp(dataset,'StackOverflow')); parameters.iter = 10;
    elseif (strcmp(dataset,'Biomedical'));    parameters.iter = 15;
    end    
    STC2_CNN(dataset,parameters,evaluteScore,randinit4Methods,randinit4KMeans,repeatNum,nbcluster);
elseif (strcmp(method,'STC2_LPI'))
    disp('Here is the proposed method: STC2_LPI!');
    fea=tf_idf(fea);
    fea = normalize(fea);
    rand('state',randinit4Methods)
    randn('state',randinit4Methods)
    disp(['state randinit4Methods:',num2str(randinit4Methods)]);
    W = constructW(fea,options);
    if (strcmp(dataset,'SearchSnippets'));    nLowVec= 20;
    elseif (strcmp(dataset,'StackOverflow')); nLowVec= 80;
    elseif (strcmp(dataset,'Biomedical'));    nLowVec= 30;
    end 
    options.PCARatio = 1;
    options.ReducedDim=nLowVec;
    [eigvector, ~] = lpp(W, options, fea);
    disp(['nLowVec is:',num2str(nLowVec)]);
    Y = fea*eigvector;
    parameters.Y = Y;
    if (strcmp(dataset,'SearchSnippets'));    parameters.iter = 15;
    elseif (strcmp(dataset,'StackOverflow')); parameters.iter = 10;
    elseif (strcmp(dataset,'Biomedical'));    parameters.iter = 15;
    end
    STC2_CNN(dataset,parameters,evaluteScore,randinit4Methods,randinit4KMeans,repeatNum,nbcluster);     
elseif (strcmp(method,'para2vecs'))
    disp(['Here is baseline clustering method: ',method]);    
    rand('state',randinit4Methods) 
    randn('state',randinit4Methods) 
    disp(['state randinit4Methods:',num2str(randinit4Methods)]);
    deep_fea_file = ['./Para2vec/',dataset,'_',method,'.mat'];
    load(deep_fea_file);
    eval(['Y=',dataset,'para2vecs;'])
    for i=1:repeatNum
        rand('state',randinit4KMeans)
        randn('state',randinit4KMeans)
        disp(['state randinit4KMeans:',num2str(randinit4KMeans)]);
        randinit4KMeans = randinit4KMeans+1;
        res = kmeans(Y, nbcluster);
        res = bestMap(gnd,res);
        %=============  evaluate AC: accuracy ========================================
        AC = length(find(gnd == res))/length(gnd)*100;
        %=============  evaluate MIhat: nomalized mutual information =================
        MIhat = MutualInfo(gnd,res)*100;
        disp(['Evalute:',num2str(i),', ACC is ',num2str(AC),' and NMI is ',num2str(MIhat)]);
        evaluteScore(index_sub,index*2+1)=AC;
        evaluteScore(index_sub,index*2+2)=MIhat;
        index_sub=index_sub+1;
    end
    disp(['Final evalute results: mean ACC is ',num2str(mean(evaluteScore(:,index*2+1))),...
                            ' and mean NMI is ',num2str(mean(evaluteScore(:,index*2+2)))]);    
elseif (strncmpi('skip_',method,5))
    disp(['Here is baseline clustering method: ',method]);    
    rand('state',randinit4Methods) 
    randn('state',randinit4Methods) 
    disp(['state randinit4Methods:',num2str(randinit4Methods)]);
    skip_vector_file = ['./skip_thought/results/',dataset,'_skip.mat'];
    load(skip_vector_file);
    eval(['Y=',dataset,'_skip;'])
    if (strcmp(method,'skip_uni'));          Y = Y(:,1:2400);
    elseif (strcmp(method,'skip_bi'));       Y = Y(:,2401:4800);
    elseif (strcmp(method,'skip_combine'));  Y = Y;
    else error(['Please input a right skip method rather than ''',method,''''])
    end
    for i=1:repeatNum
        rand('state',randinit4KMeans)
        randn('state',randinit4KMeans)
        disp(['state randinit4KMeans:',num2str(randinit4KMeans)]);
        randinit4KMeans = randinit4KMeans+1;
        res = kmeans(Y, nbcluster);
        res = bestMap(gnd,res);
        %=============  evaluate AC: accuracy ========================================
        AC = length(find(gnd == res))/length(gnd)*100;
        %=============  evaluate MIhat: nomalized mutual information =================
        MIhat = MutualInfo(gnd,res)*100;
        disp(['Evalute:',num2str(i),', ACC is ',num2str(AC),' and NMI is ',num2str(MIhat)]);
        evaluteScore(index_sub,index*2+1)=AC;
        evaluteScore(index_sub,index*2+2)=MIhat;
        index_sub=index_sub+1;
    end
    disp(['Final evalute results: mean ACC is ',num2str(mean(evaluteScore(:,index*2+1))),...
                            ' and mean NMI is ',num2str(mean(evaluteScore(:,index*2+2)))]);    

else
    error(['You input a invalid method:',method,'!'])
end