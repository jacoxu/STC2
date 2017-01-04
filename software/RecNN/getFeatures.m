function features = getFeatures(words_reIndexed,beta,We,We2,W1,W2,W3,W4,b1,b2,b3,Wcat,bcat,alpha_cat,...
   embedding_size, labels, freq, func, func_prime, training)

% flag
% 1 - top node only
% 2 - average of all nodes

num_examples = length(words_reIndexed);

features1 = zeros(num_examples, embedding_size);
features2 = zeros(num_examples, embedding_size);

features = zeros(num_examples, embedding_size, 2);


parfor ii = 1:num_examples;
    words_rI = words_reIndexed{ii};
    nn = length(words_rI);

    freq_here = freq(words_rI);

    L = We(:, words_rI);
    if training == 1
        words_embedded = We2(:, words_rI) + L;
    else
        words_embedded = L;
    end
    
    Tree = forwardPropRAE([], W1,W2,W3,W4,b1,b2,b3, Wcat, bcat, alpha_cat, 0,beta, words_embedded, labels(:,ii), ...
        embedding_size, nn, freq_here, func, func_prime);
    
    if nn>1
        features1(ii,:) = Tree.nodeFeatures(:,end)';

        tempFeatures = zeros(2*nn-1, embedding_size);
        for i=1:2*nn-1
            tempFeatures(i,:) = Tree.nodeFeatures(:,i)';
        end
        features2(ii,:) = sum(tempFeatures)/(2*nn-1);
    elseif nn==1
        features1(ii,:) = Tree.nodeFeatures(:,1);
        features2(ii,:) = Tree.nodeFeatures(:,1);
    else
        features1(ii,:) = zeros(hiddenSize,1);
        features2(ii,:) = zeros(hiddenSize,1);
    end

end
features(:,:,1) = features1;
features(:,:,2) = features2;

end