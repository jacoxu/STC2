function acc = Accuracy(X, decodeInfo, minibatch, labels, mini_msk, p)

[CR_E, CR_1, CR_1_b, CR_2, CR_2_b, CR_3, CR_3_b, CR_Z, ~, ~] = stack2param(X, decodeInfo);

CR_E(:,p(30)) = 0; %%Affects calculation of gradient (and dropout)

size_mini = length(labels);
%down
data = reshape(permute(reshape(repmat(CR_E(:,minibatch),p(3),1),p(1)*p(3),p(2),size_mini),[1,3,2]),p(3)*p(1)*size_mini,p(2));

%% Composition Maps One
kernel_one = repmat(CR_1,size_mini,1);
M_1 = fastConv(data,fliplr(kernel_one),'f',p(31));

if p(12) % Folding first layer
    %up sum down
    M_1 = reshape(permute(reshape(M_1',[],p(1),p(3)*size_mini),[2,1,3]),p(1),[]);
    p(1) = p(1)/2; %Temp  
    M_1 = M_1(1:p(1),:) + M_1(p(1)+1:end,:);
    M_1 = reshape(permute(reshape(M_1,p(1),[],p(3)*size_mini),[1,3,2]),p(3)*p(1)*size_mini,[]);     
end

%% K-Max pooling
%apply -inf mask (1)
mask_1_1 = repmat(reshape(mini_msk(:,1:p(25))',1,[]),p(1)*p(3),1);
mask_1_1 = reshape(permute(reshape(mask_1_1,p(1)*p(3),[],size_mini),[1,3,2]),p(3)*p(1)*size_mini,[]);
M_1(mask_1_1) = -Inf; 

%apply index mask (2)
[i, indx] = sort(M_1,2,'descend'); 
mask_1_2 = repmat(reshape(mini_msk(:,p(25)+1:2*p(25))',1,[]),p(1)*p(3),1);
mask_1_2 = reshape(permute(reshape(mask_1_2,p(1)*p(3),[],size_mini),[1,3,2]),p(3)*p(1)*size_mini,[]);
indx(mask_1_2) = p(25)+1; 

%sort masked indices (3)
sorted_indx = sort(indx,2);

%remask -inf to 0 and extend width by one zero column (4)
M_1(mask_1_1) = 0;
if p(31) %gpu
    M_1 = [M_1,gpuArray.zeros(size(M_1,1),1,'single')];
    sorted_indx = [sorted_indx,(p(25)+1)*gpuArray.ones(size(sorted_indx,1),1,'single')];
else
    M_1 = [M_1,zeros(size(M_1,1),1)];
    sorted_indx = [sorted_indx,(p(25)+1)*ones(size(sorted_indx,1),1)];
end

%pool elements in specified order
subs_1 = sub2ind(size(M_1),repmat((1:(p(1)*p(3)*size_mini))',p(25)+1,1), sorted_indx(:));
M_1 = reshape(M_1(subs_1),(p(1)*p(3)*size_mini),p(25)+1);

%truncate to max pooled width
M_1 = M_1(:,1:p(26)-p(6)+1);
map_wdt = p(26)-p(6)+1;
mask_1_2 = mask_1_2(:,1:map_wdt); %for backprop

%% Nonlinearity after K-Max pooling
M_1 = reshape(bsxfun(@plus,reshape(M_1,p(1)*p(3),[]),CR_1_b),p(1)*p(3)*size_mini,[]); %Apply bias on wrong reshape, but it is equivalent
M_1(mask_1_2) = 0;
if p(8)
    M_1 = max(0,M_1); %relu
else
    M_1 = tanh(M_1);
end

%Up
m_1 = reshape(permute(reshape(M_1',map_wdt,p(1)*p(3),size_mini),[2,1,3]),p(1)*p(3),map_wdt*size_mini);

if p(10)>=2 %Composition maps two
    %down
    M_1 = reshape(permute(reshape(repmat(m_1,p(5),1),p(1)*p(3)*p(5),map_wdt,size_mini),[1,3,2]),p(5)*p(3)*p(1)*size_mini,map_wdt);
    kernel_two = repmat(CR_2,size_mini,1);
    
    M_2 = fastConv(M_1,fliplr(kernel_two),'f',p(31));
    map_wdt = p(26);
    
    %up
    M_2 = reshape(permute(reshape(M_2',map_wdt,p(1)*p(3)*p(5),size_mini),[2,1,3]),p(1)*p(3),map_wdt*p(5)*size_mini);
    %sum maps within second conv layer
    M_2 = reshape(sum(reshape(M_2',[],p(3)),2),[],p(1))';
    %down
    M_2 = reshape(permute(reshape(M_2,p(1)*p(5),map_wdt,size_mini),[1,3,2]),p(5)*p(1)*size_mini,map_wdt);
    
    if p(13) % Folding second layer
        %up sum down
        M_2 = reshape(permute(reshape(M_2',[],p(1),p(5)*size_mini),[2,1,3]),p(1),[]);
        p(1) = p(1)/2; %
        
        M_2 = M_2(1:p(1),:) + M_2(p(1)+1:end,:);
        M_2 = reshape(permute(reshape(M_2,p(1),[],p(5)*size_mini),[1,3,2]),p(5)*p(1)*size_mini,[]);
    end
    
    %% K-Max pooling -- 2nd layer --
    %apply -inf mask (1)
    mask_2_1 = repmat(reshape(mini_msk(:,2*p(25)+1:2*p(25)+p(26))',1,[]),p(1)*p(5),1);
    mask_2_1 = reshape(permute(reshape(mask_2_1,p(1)*p(5),[],size_mini),[1,3,2]),p(5)*p(1)*size_mini,[]);
    M_2(mask_2_1) = -Inf; 

    %apply index mask (2)
    [i, indx] = sort(M_2,2,'descend'); 
    mask_2_2 = repmat(reshape(mini_msk(:,2*p(25)+p(26)+1:2*p(25)+2*p(26))',1,[]),p(1)*p(5),1);
    mask_2_2 = reshape(permute(reshape(mask_2_2,p(1)*p(5),[],size_mini),[1,3,2]),p(5)*p(1)*size_mini,[]);
    indx(mask_2_2) = p(26)+1; 

    %sort masked indices (3)
    sorted_indx_2 = sort(indx,2);

    %remask -inf to 0 and extend width by one zero column (4)
    M_2(mask_2_1) = 0;
    if p(31) %gpu
        M_2 = [M_2,gpuArray.zeros(size(M_2,1),1,'single')];
        sorted_indx_2 = [sorted_indx_2,(p(26)+1)*gpuArray.ones(size(sorted_indx_2,1),1,'single')];
    else
        M_2 = [M_2,zeros(size(M_2,1),1)];
        sorted_indx_2 = [sorted_indx_2,(p(26)+1)*ones(size(sorted_indx_2,1),1)];
    end

    %pool elements in specified order
    subs_2 = sub2ind(size(M_2),repmat((1:(p(1)*p(5)*size_mini))',p(26)+1,1), sorted_indx_2(:));
    M_2 = reshape(M_2(subs_2),(p(1)*p(5)*size_mini),p(26)+1);

    %truncate to max pooled width
    M_2 = M_2(:,1:p(27)-p(36)+1);
    map_wdt = p(27)-p(36)+1;
    mask_2_2 = mask_2_2(:,1:map_wdt); %truncate mask for use in backprop
    
    %% Nonlinearity after K-Max pooling
    M_2 = reshape(bsxfun(@plus,reshape(M_2,p(1)*p(5),[]),CR_2_b),p(1)*p(5)*size_mini,[]); %Apply bias on wrong reshape, but it is equivalent
    M_2(mask_2_2) = 0;
    if p(8)
        M_2 = max(0,M_2); %relu
    else
        M_2 = tanh(M_2);
    end

    %Up
    m_2 = reshape(permute(reshape(M_2',map_wdt,p(1)*p(5),size_mini),[2,1,3]),p(1)*p(5),map_wdt*size_mini);
    
    if p(10) == 3 %% third layer
        
        %repeat for multiple maps in third layer
        M_2 = reshape(permute(reshape(repmat(m_2,p(37),1),p(1)*p(5)*p(37),map_wdt,size_mini),[1,3,2]),p(37)*p(5)*p(1)*size_mini,map_wdt);
        kernel_three = repmat(CR_3,size_mini,1);
        
        M_3 = fastConv(M_2,fliplr(kernel_three),'f',p(31));
        map_wdt = p(27);
        
        %up
        M_3 = reshape(permute(reshape(M_3',map_wdt,p(1)*p(5)*p(37),size_mini),[2,1,3]),p(1)*p(5),map_wdt*p(37)*size_mini);
        %sum maps within second conv layer
        M_3 = reshape(sum(reshape(M_3',[],p(5)),2),[],p(1))';
        %down
        M_3 = reshape(permute(reshape(M_3,p(1)*p(37),map_wdt,size_mini),[1,3,2]),p(37)*p(1)*size_mini,map_wdt);
        
        if p(35) % Folding third layer
            %up sum down
            M_3 = reshape(permute(reshape(M_3',[],p(1),p(37)*size_mini),[2,1,3]),p(1),[]);
            p(1) = p(1)/2; %
            
            M_3 = M_3(1:p(1),:) + M_3(p(1)+1:end,:);
            M_3 = reshape(permute(reshape(M_3,p(1),[],p(37)*size_mini),[1,3,2]),p(37)*p(1)*size_mini,[]);
        end
        
       %% K-Max pooling -- 3rd layer --
        %apply -inf mask (1)
        mask_3_1 = repmat(reshape(mini_msk(:,2*p(25)+2*p(26)+1:2*p(25)+2*p(26)+p(27))',1,[]),p(1)*p(37),1);
        mask_3_1 = reshape(permute(reshape(mask_3_1,p(1)*p(37),[],size_mini),[1,3,2]),p(37)*p(1)*size_mini,[]);
        M_3(mask_3_1) = -Inf;
        
        %apply index mask (2)
        [i, indx] = sort(M_3,2,'descend');
        mask_3_2 = repmat(reshape(mini_msk(:,2*p(25)+2*p(26)+p(27)+1:end)',1,[]),p(1)*p(37),1);
        mask_3_2 = reshape(permute(reshape(mask_3_2,p(1)*p(37),[],size_mini),[1,3,2]),p(37)*p(1)*size_mini,[]);
        indx(mask_3_2) = p(27)+1;
        
        %sort masked indices (3)
        sorted_indx_3 = sort(indx,2);
        
        %remask -inf to 0 and extend width by one zero column (4)
        M_3(mask_3_1) = 0;
        if p(31) %gpu
            M_3 = [M_3,gpuArray.zeros(size(M_3,1),1,'single')];
            sorted_indx_3 = [sorted_indx_3,(p(27)+1)*gpuArray.ones(size(sorted_indx_3,1),1,'single')];
        else
            M_3 = [M_3,zeros(size(M_3,1),1)];
            sorted_indx_3 = [sorted_indx_3,(p(27)+1)*ones(size(sorted_indx_3,1),1)];
        end
        
        %pool elements in specified order
        subs_3 = sub2ind(size(M_3),repmat((1:(p(1)*p(37)*size_mini))',p(27)+1,1), sorted_indx_3(:));
        M_3 = reshape(M_3(subs_3),(p(1)*p(37)*size_mini),p(27)+1);
        
        %truncate to max pooled width
        M_3 = M_3(:,1:p(7));
        map_wdt = p(7);
        %mask_3_2 = mask_3_2(:,1:map_wdt); not needed
        
        %% Nonlinearity after K-Max pooling
        M_3 = reshape(bsxfun(@plus,reshape(M_3,p(1)*p(37),[]),CR_3_b),p(1)*p(37)*size_mini,[]); %Apply bias on wrong reshape, but it is equivalent
        %M_3(mask_3_2) = 0; trivial
        if p(8)
            M_3 = max(0,M_3); %relu
        else
            M_3 = tanh(M_3);
        end
        
        %Up and vectorize for classification
        M_3 = reshape(permute(reshape(M_3',map_wdt,p(1)*p(37),size_mini),[2,1,3]),p(1)*p(37),map_wdt*size_mini);
        M_3 = reshape(M_3,p(1)*p(37)*map_wdt,size_mini);
        
    else
        %vectorize two-layer model for classification
        M_3 = reshape(m_2,p(1)*p(5)*map_wdt,size_mini); 
    end
else
    %vectorize one-layer model for classification
    M_3 = reshape(m_1,p(1)*p(3)*map_wdt,size_mini);
end

%%Classification
if p(31) %if GPU
    b_w = gpuArray.ones(1,size_mini,'single');
else
    b_w = ones(1,size_mini);
end

if p(40)
    M_3 = M_3./2;
end

Z = exp(CR_Z*[M_3;b_w]);
Z = bsxfun(@rdivide,Z,sum(Z));

[val,ind] = max(Z);
acc = sum(ind' == labels);
end

