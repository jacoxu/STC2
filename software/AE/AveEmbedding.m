function Y=AveEmbedding(dataset,parameters)
dataStr=['./../dataset/',dataset,'-STC2.mat'];
load(dataStr);
parameters.vocSize = size_vocab;
% Step 1. Generate word vector sets
CR_E = randi([-25,25],parameters.wordDim,parameters.vocSize)/100;
disp(strcat('Number of weights E:',num2str(size(CR_E))));
vocab_emb_length = length(vocab_emb_Word2vec_48(1,:));
if vocab_emb_length > size_vocab
    error(['Error, and the size fo vocab_emb is:',vocab_emb_length])
end
CR_E(1:parameters.wordDim,vocab_emb_Word2vec_48_index) = vocab_emb_Word2vec_48(1:parameters.wordDim,1:vocab_emb_length);
% Step 2. Compute TF-IDF
if parameters.weightMode
    fea_All=tf_idf(fea_All);
end
% Step 3. Average Embedding
textSize = length(fea_All(:,1));
fea_vector =[];
for i=1:textSize
    tmp_fea_vector_weight = repmat(fea_All(i,find(fea_All(i,:)>0)),parameters.wordDim,1);
    tmp_fea_vector_matrix = CR_E(:,find(fea_All(i,:)>0)) .* tmp_fea_vector_weight;
    tmp_fea_vector = sum(tmp_fea_vector_matrix,2);
    fea_vector(i,:) = tmp_fea_vector';
end
% Step 4. Normalize features
Y = normalize(fea_vector);
end