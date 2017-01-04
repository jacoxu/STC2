%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We modified the code from
% Semi-Supervised Recursive Autoencoders for Predicting Sentiment Distributions
% Richard Socher, Jeffrey Pennington, Eric Huang, Andrew Y. Ng, and Christopher D. Manning
% Conference on Empirical Methods in Natural Language Processing (EMNLP 2011)
% See http://www.socher.org for more information or to ask questions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Y=RecNN(randinit4Methods,randinit4KMeans,dataset,parameters)
data_f=['./../dataset/',dataset,'-STC2.mat'];
if (strcmp(dataset,'SearchSnippets'));     options.maxIter = 8;
elseif (strcmp(dataset,'StackOverflow'));  options.maxIter = 6;
elseif (strcmp(dataset,'Biomedical'));     options.maxIter = 6;
end
rand('state',randinit4Methods) 
randn('state',randinit4Methods)
disp(['state randinit4KMeans:',num2str(randinit4Methods)]);
params.randinit4KMeans = randinit4KMeans;
% load minFunc
addpath(genpath('tools/'))

%%%%%%%%%%%%%%%%%%%%%%
% Hyperparameters
%%%%%%%%%%%%%%%%%%%%%%
% set this to 1 to train the model and to 0 for just testing the RAE features (and directly training the classifier)
params.trainModel = 1; %1: need to train model, 0: directly to test

% node and word size
params.embedding_size = parameters.wordDim; %set the size of WordEmbedding

% Relative weighting of reconstruction error and categorization error
params.alpha_cat = 0.2; 

% Regularization: lambda = [lambdaW, lambdaL, lambdaCat, lambdaLRAE];
params.lambda = [1e-05, 0.0001, 1e-07, 0.01]; 

% weight of classifier cost on nonterminals
params.beta=0.5;

func = @norm1tanh;
func_prime = @norm1tanh_prime;

% parameters for the optimizer
options.Method = 'lbfgs';
options.display = 'on';

disp(params);
disp(options);

%%%%%%%%%%%%%%%%%%%%%%
% Pre-process dataset 
%%%%%%%%%%%%%%%%%%%%%%
% set this to different folds (1-10) and average to reproduce the results in the paper
params.CVNUM = 1;

% read in polarity dataset
if ~exist(data_f,'file')
    read_rtPolarity
else
    load(data_f,'all','vocab_emb_Word2vec_48*','size_vocab','labels_All');
end

parameters.vocSize = size_vocab-1;
% Step 1: generate word vectors based on the vocabulary and trained word vectors.
We2 = randi([-25,25],parameters.wordDim,parameters.vocSize)/100; %first create one random matrix for word vector
vocab_emb_length = length(vocab_emb_Word2vec_48(1,:));
% put the trained vectors on right position
if vocab_emb_length > size_vocab
    error(['Error, and the size fo vocab_emb is:',vocab_emb_length])
end
We2(1:parameters.wordDim,vocab_emb_Word2vec_48_index) = vocab_emb_Word2vec_48(1:parameters.wordDim,1:vocab_emb_length);
for i=1:length(labels_All)
    tmpSNum = all(i,:);
    tmpSNum(tmpSNum==size_vocab) =[];
    allSNum{i} = tmpSNum;
end
allSNum = allSNum';
labels=labels_All';
clear vocab_emb_Word2vec_48* size_vocab labels_All;

% Greedy Unsupervised RAE
sent_freq = ones(length(allSNum),1);
[~,dictionary_length] = size(We2); %We2 is word2vec

% split this current fold into train and test
index_list_train = cell2mat(allSNum'); %read all index of trained corpus
unq_train = sort(index_list_train); %sort the index
freq_train = histc(index_list_train,1:size(We2,2)); %static the frequency of each word.
freq_train = freq_train/sum(freq_train); %norm the frequency

cat_size=1;% for multinomial distributions this would be >1
numExamples = length(allSNum); % Size of the corpus

%%%%%%%%%%%%%%%%%%%%%%
% Initialize parameters, then put them all into parameter Theta.
%%%%%%%%%%%%%%%%%%%%%%
theta = initializeParameters(params.embedding_size, params.embedding_size, cat_size, dictionary_length);

%%%%%%%%%%%%%%%%%%%%%%
% Train Model
%%%%%%%%%%%%%%%%%%%%%%

sent_freq_here = sent_freq(1:numExamples);  % get the sentence num. of each text, althought they are ones.
% start to train and find the optimal parameters
[~, ~,Y] = minFunc( @(p)RAECost(p, params.alpha_cat, cat_size,params.beta, dictionary_length, params.embedding_size, ...
    params.lambda, We2, allSNum, labels, freq_train, sent_freq, func, func_prime), ...
    theta, options,...
    params, allSNum,labels, cat_size, dictionary_length,freq_train, func, func_prime,We2);
disp('RecNN is done, OK!')

end
