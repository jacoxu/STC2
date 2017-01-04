clear; clc;
dataset='SearchSnippets'; %SearchSnippets, StackOverflow, Biomedical
method = 'SVM'; % SVM(Support Vector Machine)
isKernel = 0; %1: SVM Gaussian Kernel; 0: SVM Linear Kernel.
Weighting = 'TF'; %TF, TFIDF or AE(Average Embedding)
dataStr=['./../../dataset/',dataset,'-lite.mat'];
load(dataStr);
rand('state',0)
randn('state',0)
%%
disp('Step 1 get the train and test data ...')
% TFIDF
if (strcmp(Weighting,'TF'))
    testFea = fea(testIdx, :);
    trainFea = fea(trainIdx, :);
    testGnd = gnd(testIdx, :);
    trainGnd = gnd(trainIdx, :);
elseif (strcmp(Weighting,'TFIDF'))
    testFea = fea(testIdx, :);
    trainFea = fea(trainIdx, :);
    testGnd = gnd(testIdx, :);
    trainGnd = gnd(trainIdx, :);    
    [trainFea, testFea] = tf_idf(trainFea, testFea);
elseif (strcmp(Weighting,'AE'))
    dataStr=['./../../dataset/',dataset,'-STC2.mat'];
    load(dataStr);
    parameters.wordDim = 48;
    parameters.vocSize = size_vocab;
    disp('AE: Generate verage embedding vectors ...')
    % Step a. Generate word vector sets
    CR_E = randi([-25,25],parameters.wordDim,parameters.vocSize)/100;
    disp(strcat('Number of weights E:',num2str(size(CR_E))));
    vocab_emb_length = length(vocab_emb_Word2vec_48(1,:));
    if vocab_emb_length > size_vocab
        error(['Error, and the size fo vocab_emb is:',vocab_emb_length])
    end
    CR_E(1:parameters.wordDim,vocab_emb_Word2vec_48_index) = vocab_emb_Word2vec_48(1:parameters.wordDim,1:vocab_emb_length);
    % Step b. Average Embedding
    textSize = length(fea_All(:,1));
    fea =[];
    for i=1:textSize
        tmp_fea_vector_weight = repmat(fea_All(i,find(fea_All(i,:)>0)),parameters.wordDim,1);
        tmp_fea_vector_matrix = CR_E(:,find(fea_All(i,:)>0)) .* tmp_fea_vector_weight;
        tmp_fea_vector = sum(tmp_fea_vector_matrix,2);
        fea(i,:) = tmp_fea_vector';
        if mod(i,2000) == 0
            disp(['has averaged embedding number:',num2str(i)]);
        end
    end
    testFea = fea(testIdx, :);
    trainFea = fea(trainIdx, :);
    testGnd = gnd(testIdx, :);
    trainGnd = gnd(trainIdx, :);
    %
end
testFea = normalize(testFea);
trainFea = normalize(trainFea);
%%
disp('step 2 train model ...')
if (strcmp(method,'SVM'))
    trainFea = sparse(trainFea);
    testFea = sparse(testFea);
    if ~isKernel
        disp('start train linear SVM model ...')
        model = train(trainGnd, trainFea, '-q');
        disp('start predict test data via linear SVM ...')
        disp('step 3 predict test data ...')
        [predict_label, accuracy, predict_scores] = predict(testGnd, testFea, model, '-b 1');
    else
        disp('start train kernel SVM model ...')
        model = svmtrain(trainGnd, trainFea, '-t 0');
        disp('start predict test data via kernel SVM model ...')
        disp('step 3 predict test data ...')
        [predict_label, accuracy, predict_scores] = svmpredict(testGnd, testFea, model);
    end
end
AC = length(find(predict_label == testGnd))/length(testGnd)*100;
disp(['Accuracy is ',num2str(AC)])

