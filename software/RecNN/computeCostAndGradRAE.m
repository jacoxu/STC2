function [cost_total,grad_total, allKids] = computeCostAndGradRAE(allKids, theta, updateWcat, alpha_cat, cat_size, beta, ...
    dictionary_length,  hiddenSize, lambda, We_orig, data_cell, labels, freq_orig, f, f_prime)

lambdaW = lambda(1);
lambdaL = lambda(2);
lambdaCat = lambda(3);

num_examples = length(data_cell);
labels2 = labels;

if isempty(allKids)
    allKids = cell(num_examples,1);
end

[W1, W2, W3, W4, b1, b2, b3, Wcat, bcat, We] = getW(updateWcat, theta, hiddenSize, cat_size, dictionary_length);

cost_total = 0;
gradW1 = zeros(size(W1));
gradW2 = zeros(size(W2));
gradW3 = zeros(size(W3));
gradW4 = zeros(size(W4));
gradb1 = zeros(size(b1));
gradb2 = zeros(size(b2));
gradb3 = zeros(size(b3));
gradbcat = zeros(size(bcat));
gradWcat = zeros(size(Wcat));
grad_We_total = sparse(hiddenSize, dictionary_length);
num_nodes = 0;

range = 1:num_examples;
parfor ii = range;
    data = data_cell{ii};
    
    words_indexed = data;
    L = We(:, words_indexed);
    grad_We = sparse(hiddenSize, dictionary_length);
    words_embedded = We_orig(:, words_indexed) + L;
    gradL = zeros(size(L));
    current_label = labels2(:,ii);
    
    
    freq = freq_orig(words_indexed);
    
    [~, sl] = size(words_embedded);
    if sl > 1 %don't process one word sentences
        
        % Forward Propagation
        if updateWcat
            Tree = forwardPropRAE(allKids{ii}, W1,W2,W3,W4,b1,b2,b3, Wcat, bcat, alpha_cat, updateWcat, beta, words_embedded, current_label, hiddenSize, sl, freq, f, f_prime);
            cost = sum(Tree.nodeScores) ; % Include leaf node error (supervised)
            num_nodes = num_nodes + 1;
            
        else
            Tree = forwardPropRAE([], W1,W2,W3,W4,b1,b2,b3, Wcat, bcat, alpha_cat, updateWcat,beta, words_embedded, labels, hiddenSize, sl, freq, f, f_prime);
            allKids{ii} = Tree.kids;
            cost = sum(Tree.nodeScores(sl+1:end)) ;
            num_nodes = num_nodes + sl;
        end
        
        %Backprop
        toPopulate = [2*sl-1;1;1];
        nodeFeatures = Tree.nodeFeatures;
        nodeFeatures_unnormalized = Tree.nodeFeatures_unnormalized;
        W0 = zeros(size(W1));
        W = zeros(hiddenSize,hiddenSize,3);
        W(:,:,1) = W0;
        W(:,:,2) = W1;
        W(:,:,3) = W2;
        DEL = {zeros(hiddenSize,1) Tree.node_y1c1 Tree.node_y2c2};
        
        while ~isempty(toPopulate)
            parentNode = toPopulate(:,1);
            mat = W(:,:,parentNode(2));
            del = DEL{parentNode(2)}(:,parentNode(3));
            
            if parentNode(1)>sl % Non-leaf?
                
                kids = Tree.kids(parentNode(1),:);
                kid1 = [kids(1); 2; parentNode(1)];
                kid2 = [kids(2); 3; parentNode(1)];
                
                toPopulate = [kid1 kid2 toPopulate(:, 2:end)];
                a1_unnormalized = nodeFeatures_unnormalized(:,parentNode(1));
                a1 = nodeFeatures(:,parentNode(1));
                
                nd1 = Tree.nodeDelta_out1(:,parentNode(1));
                nd2 = Tree.nodeDelta_out2(:,parentNode(1));
                pd = Tree.parentDelta(:,parentNode(1));
                
                
                if updateWcat
                    smd = Tree.catDelta(:,parentNode(1));
                    gradbcat =gradbcat + smd;
                    parent_d = f_prime(a1_unnormalized) * (W3'*nd1 + W4'*nd2 + mat'*pd + Wcat'*smd - del);
                    gradWcat = gradWcat + smd*a1';
                    
                else
                    parent_d = f_prime(a1_unnormalized) * (W3'*nd1 + W4'*nd2 + mat'*pd - del);
                end
                
                gradb1 = gradb1 + parent_d;
                gradb2 = gradb2 + nd1;
                gradb3 = gradb3 + nd2;
                
                Tree.parentDelta(:,toPopulate(1,1)) = parent_d;
                
                Tree.parentDelta(:,toPopulate(1,2)) = parent_d;
                
                gradW1 = gradW1 + parent_d*nodeFeatures(:,toPopulate(1,1))';
                gradW2 = gradW2 + parent_d*nodeFeatures(:,toPopulate(1,2))';
                gradW3 = gradW3 + nd1*a1';
                gradW4 = gradW4 + nd2*a1';
                
            else % leaf
                if updateWcat
                    gradWcat = gradWcat + Tree.catDelta(:, parentNode(1)) * nodeFeatures(:,parentNode(1))';
                    gradbcat = gradbcat + Tree.catDelta(:,parentNode(1));
                    gradL(:,toPopulate(1,1)) = gradL(:,toPopulate(1,1)) + ...
                        (mat'*Tree.parentDelta(:,toPopulate(1,1)) + Wcat'*Tree.catDelta(:,toPopulate(1,1)) - del);
                else
                    gradL(:,toPopulate(1,1)) = gradL(:,toPopulate(1,1)) + ...
                        (mat'*Tree.parentDelta(:,toPopulate(1,1)) - del);
                    
                end
                toPopulate(:,1) = [];
                
            end
        end
        
        for l=1:sl
            grad_We(:, words_indexed(l)) = grad_We(:, words_indexed(l)) + gradL(:,l);
        end
        grad_We_total = grad_We_total + grad_We;
        
        cost_total = cost_total + cost;
    else
        num_nodes = num_nodes+1;
    end
    
end

if ~updateWcat
    num_nodes = num_nodes - length(range);
end

cost_total = 1/num_nodes*cost_total + lambdaW/2 * ( W1(:)'*W1(:) + W2(:)'*W2(:) + W3(:)'*W3(:) + W4(:)'*W4(:) );

grad_total = 1/num_nodes*[gradW1(:); gradW2(:); gradW3(:); gradW4(:); gradb1; gradb2; gradb3] ...
    + [lambdaW*W1(:); lambdaW*W2(:); lambdaW*W3(:); lambdaW*W4(:); zeros(3*hiddenSize,1)];
    
if updateWcat
    gradWcat = 1/num_nodes *  [gradWcat(:); gradbcat] + [lambdaCat * Wcat(:); zeros(size(bcat))];
    cost_total = cost_total + lambdaCat/2 * Wcat(:)'*Wcat(:);
    grad_total = [grad_total(:);  gradWcat(:)];
end

cost_total = cost_total +  lambdaL/2 * We(:)'*We(:);
grad_total = [grad_total(:); 1/num_nodes*grad_We_total(:) + lambdaL * We(:)];

