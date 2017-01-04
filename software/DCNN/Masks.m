function [train_msk, p] = Masks(train, train_lbl, p)
%% Matrices easying the vectorized computation in CostFunction

%% Two types of masks: one is to mask with -Inf values that are superfluous due to sentence length overhead
%% the second is to apply differentiated k-max pooling that depends on the length of each sentene; turning superfluous values to max length of first layer

%Lengths of mask independenct of tot_num_layers
p(25) = p(2)+p(4)-1; %len of masks length+pool layer 1 
p(26) = indexMaskLen(p(2), 1, p) + p(6)-1; %len of masks length+pool layer 2 
p(27) = indexMaskLen(p(2), 2, p) + p(36)-1; %len of masks length+pool layer 3 

if p(10) == 1 %If one layer in total 
    train_msk=zeros(size(train,1), 2*p(25)); 
    
    train_msk = findMasks(train, train_lbl, train_msk, p);
elseif p(10) == 2 
    train_msk=zeros(size(train,1), 2*p(25)+2*p(26)); 
    
    train_msk = findMasks(train, train_lbl, train_msk, p);
    
elseif p(10) == 3 
    train_msk=zeros(size(train,1), 2*p(25)+2*p(26)+2*p(27)); 
    train_msk = findMasks(train, train_lbl, train_msk, p);
    
end

train_msk = logical(train_msk);

end

function train_msk = findMasks(train, train_lbl, train_msk ,p)
for i=1:length(train)
    if p(60)==1
        sent_len = train_lbl(i,2); %contains length 
    else
        sent_len = train_lbl{2}(i); %contains length 
    end
    
    if p(10) == 1 
        train_msk(i,sent_len+p(4)-1+1:p(25)) = 1; %layer 1 sent_lengths mask
        train_msk(i,p(25)+p(7)+1:end) = 1; %layer 1 max_pool mask
    elseif p(10) == 2
        train_msk(i,sent_len+p(4)-1+1:p(25)) = 1; %layer 1 sent_lengths mask
        train_msk(i,p(25)+indexMaskLen(sent_len,1,p)+1:2*p(25)) = 1; %layer 1 max_pool mask
        
        train_msk(i,2*p(25)+indexMaskLen(sent_len,1,p)+p(6)-1+1:2*p(25)+p(26)) = 1; %layer 2 sent_lengths mask
        train_msk(i,2*p(25)+p(26)+p(7)+1:end) = 1; %layer 2 max_pool mask
        
    elseif p(10) == 3
        train_msk(i,sent_len+p(4)-1+1:p(25)) = 1; %layer 1 sent_lengths mask
        train_msk(i,p(25)+indexMaskLen(sent_len,1,p)+1:2*p(25)) = 1; %layer 1 max_pool mask
        
        train_msk(i,2*p(25)+indexMaskLen(sent_len,1,p)+p(6)-1+1:2*p(25)+p(26)) = 1; %layer 2 sent_lengths mask
        train_msk(i,2*p(25)+p(26)+indexMaskLen(sent_len,2,p)+1:2*p(25)+2*p(26)) = 1; %layer 2 max_pool mask
        
        train_msk(i,2*p(25)+2*p(26)+indexMaskLen(sent_len,2,p)+p(36)-1+1:2*p(25)+2*p(26)+p(27)) = 1; %layer 3 sent_lengths mask
        train_msk(i,2*p(25)+2*p(26)+p(27)+p(7)+1:end) = 1; %layer 3 max_pool mask
        
    end
end

end


function pool_size = indexMaskLen(sent_len, num_conv_layer, p)
%Function computes pooling size for a given layer
if p(10) == 1
    pool_size = p(7); %size of final pooling layer
    
elseif p(10) == 2
    if num_conv_layer == 1
        pool_size = max(ceil(sent_len/2),p(7)); %half if shallow network
    end
    
    if num_conv_layer == 2 %not in fact possible
        pool_size = p(7);
    end
elseif p(10) == 3
    if num_conv_layer == 1
        pool_size = max(ceil((sent_len/3)*2),p(7)); %linear function 
    end
    
    if num_conv_layer == 2
        pool_size = max(ceil(sent_len/3),p(7));
    end
else
    pool_size = -1;
end

end

