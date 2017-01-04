function [cost,grad] = CostFunction(X, decodeInfo, minibatch, labels, mini_msk, indices, p)

%PARAMS/OPTIONS
% p(1) = 42;          disp(strcat('Size word vectors:',num2str(p(1))));
% p(2) = sent_length; disp(strcat('Max sent length:',num2str(p(2))));
% p(3) = 5;           disp(strcat('Number feat maps in first layer:',num2str(p(3))));
% p(5) = 10;           disp(strcat('Number feat maps in second layer:', num2str(p(5))));
% p(37) = 18;          disp(strcat('Number feat maps in third layer:', num2str(p(37))));
% p(4) = 6;           disp(strcat('Size of kernel in first layer:', num2str(p(4))));
% p(6) = 5;           disp(strcat('Size of kernel in second layer:', num2str(p(6))));
% p(36) = 3;          disp(strcat('Size of kernel in third layer:', num2str(p(36))));
% p(8) = 0;           disp(strcat('Using relu:',num2str(p(8))));
% p(9) = 6;           disp(strcat('Number of output classes:',num2str(p(9))));
% p(10) = 1;          disp(strcat('Number of conv layers being used (1 or 2 or 3):',num2str(p(10))));
% p(7) = 4;           disp(strcat('TOP POOLING width:',num2str(p(7)))); 
% p(12) = 1;          disp(strcat('Folding in first layer:', num2str(p(12))));
% p(13) = 1;          disp(strcat('Folding in second layer:', num2str(p(13))));
% p(35) = 0;          disp(strcat('Folding in third layer:',num2str(p(35))));
% p(30) = size_vocab; disp(strcat('Size vocab (and pad):',num2str(p(30))));
% p(32) = 1;          disp(strcat('Word embedding learning ON:',num2str(p(32))));
% p(33) = 199;        disp(strcat('if emb learn ON, after how many epochs OFF:',num2str(p(33))));
% p(34) = 1;          disp(strcat('use preinitialized vocabulary:',num2str(p(34))));
% 
% p(40) = 1;          disp(strcat('Dropout at Projection Sentence matrix:',num2str(p(40))));
% p(41) = 1;          disp(strcat('Dropout at First layer:',num2str(p(40))));
% p(42) = 1;          disp(strcat('Dropout at Second layer:',num2str(p(40))));
% p(43) = 1;          disp(strcat('Dropout at Third layer:',num2str(p(40))));
% %
% %
% disp(' ');
% p(20) = 1e-4;       disp(strcat('Reg E (word vectors):',num2str(p(20))));
% p(21) = 3e-5;       disp(strcat('Reg 1 (first conv layer):',num2str(p(21))));
% p(22) = 3e-6;       disp(strcat('Reg 2 (second conv layer):',num2str(p(22))));
% p(23) = 1e-5;       disp(strcat('Reg 3 (third conv layer):',num2str(p(23))));
% p(24) = 1e-4;       disp(strcat('Reg Z (classification layer):',num2str(p(24))));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
factor_b = p(65);
[CR_E, CR_1, CR_1_b, CR_2, CR_2_b, CR_3, CR_3_b, CR_Z] = stack2param(X, decodeInfo);
CR_E(:,p(30)) = 0; %this and dropout and fft will affect gradient check
if p(60)==1
    size_mini = length(labels); 
else
    size_mini = length(labels(:,1)); 
end

sentence_mats = CR_E(:,minibatch);

dim_length = p(1); 

if p(40) %dropout at projection layer
   DRO_0 = logical(round(rand(size(sentence_mats))));
   if p(31) %gpu
        DRO_0 = gpuArray(DRO_0);
   end
   sentence_mats = sentence_mats.*DRO_0;
end

%down
data = reshape(permute(reshape(repmat(sentence_mats,p(3),1),p(1)*p(3),p(2),size_mini),[1,3,2]),p(3)*p(1)*size_mini,p(2));

%% 
kernel_one = repmat(CR_1,size_mini,1);
% 
M_1 = fastConv(data,fliplr(kernel_one),'f',p(31));

if p(12) % Folding first layer --> Collapsing.
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
[i, indx] = sort(M_1, 2, 'descend'); 
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

if p(41) %dropout at first layer
   DRO_1 = logical(round(rand(size(m_1))));
   if p(31) %gpu
        DRO_1 = gpuArray(DRO_1);
   end
   m_1 = m_1.*DRO_1;
end  

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
        %M_2 = M_2(1:2:end,:)+M_2(2:2:end,:); %换一种Folding 方式
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

    if p(42) %dropout at second layer
        DRO_2 = logical(round(rand(size(m_2))));
        if p(31) %gpu
            DRO_2 = gpuArray(DRO_2);
        end
        m_2 = m_2.*DRO_2;
    end
    
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
            %M_3 = M_3(1:2:end,:)+M_3(2:2:end,:); %换一种Folding 方式
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
        
        %Up
        M_3 = reshape(permute(reshape(M_3',map_wdt,p(1)*p(37),size_mini),[2,1,3]),p(1)*p(37),map_wdt*size_mini);
        if p(43) %dropout at third layer
            DRO_3 = logical(round(rand(size(M_3))));
            if p(31) %gpu
                DRO_3 = gpuArray(DRO_3);
            end
            M_3 = M_3.*DRO_3;
        end
        %% vectorize three-layer model for classification
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
    b_w = factor_b.*ones(1,size_mini);
end
    
%% Logistic
Z_tmp = CR_Z*[M_3;b_w];
Z = exp(Z_tmp);
Z = Z./(1+Z); 
%cost = sum(sum(labels'.*log(Z))+ sum((1-labels').*log(1-Z)));

Z_sub2ind = (labels'==1);    
Z_sub2ind0 = (labels'==0); 
cost = sum(log(Z(Z_sub2ind)))+ sum(log(1-Z(Z_sub2ind0)));
D_Z = labels' - Z; % D_Z = y-Z


%%
Z_df = D_Z*[M_3;b_w]';

d_z = CR_Z'*D_Z; 
D_p = d_z(1:end-1,:);

%Nonlinearity top layer
%no mask
if p(8) 
    D_p = D_p.*(M_3>0);
else 
    D_p = D_p.*(1-M_3.^2);
end

%%Backprop convolutional layer
if p(10)==3 %if three layers
    
    D_p = reshape(D_p, p(1)*p(37), map_wdt*size_mini); %
    if p(43)
        D_p = D_p.*DRO_3; 
    end
    CR_3_b_df = sum(D_p,2); 
    %Down
    D_p = reshape(permute(reshape(D_p,p(1)*p(37),map_wdt,size_mini),[1,3,2]),p(1)*p(37)*size_mini,map_wdt);
    
    %% Backprop K-Max Pooling
    if p(31) %if GPU
        D_p = gather(D_p);
        subs_3 = gather(subs_3);
        q = round(gather(p));
        D_p_K = zeros(q(1)*q(37)*size_mini,q(27)+1,'single');
        D_p_k = zeros(size(D_p_K),'single');
        
        D_p_k(:,1:map_wdt) = D_p; %extend error with trailing zeros
        D_p_K(subs_3) = D_p_k;
        D_p_K = gpuArray(D_p_K(:,1:end-1));
    else
        D_p_K = zeros(p(1)*p(37)*size_mini,p(27)+1);
        D_p_k = zeros(size(D_p_K));
        
        D_p_k(:,1:map_wdt) = D_p; %extend error with trailing zeros
        D_p_K(subs_3) = D_p_k;
        D_p_K = D_p_K(:,1:end-1);
    end
    map_wdt = p(27);    
    
    if p(35) %Backprop folding
        %up
        D_p_K = reshape(permute(reshape(D_p_K',map_wdt,p(1),p(37)*size_mini),[2,1,3]),p(1),[]);
        
        D_p_K = repmat(D_p_K,2,1);
        p(1) = p(1)*2;
        %down
        D_p_K = reshape(permute(reshape(D_p_K,p(1),[],p(37)*size_mini),[1,3,2]),p(37)*p(1)*size_mini,[]);
    end
    
    
    %up
    D_p_K = repmat(reshape(permute(reshape(D_p_K',map_wdt,p(1)*p(37),size_mini),[2,1,3]),p(1),map_wdt*p(37)*size_mini),p(5),1);
    %down
    D_p_K = reshape(permute(reshape(D_p_K,p(1)*p(5)*p(37),map_wdt,size_mini),[1,3,2]),p(1)*p(5)*p(37)*size_mini,map_wdt);
    
    %compute and up
    f = reshape(permute(reshape(fastConv(fliplr(D_p_K),M_2,'v',p(31))',p(36),p(1)*p(5)*p(37),size_mini),[2,1,3]),p(1)*p(5)*p(37),p(36)*size_mini);
    CR_3_df = reshape(sum(reshape(f,p(1)*p(5)*p(37)*p(36),size_mini),2),p(1)*p(5)*p(37),p(36));

    D_p = fastConv(D_p_K, kernel_three, 'v', p(31)); 
    map_wdt = size(D_p,2);
    
    %Up and sum
    D_p = reshape(permute(reshape(D_p',map_wdt,p(1)*p(5)*p(37),size_mini),[2,1,3]),p(1)*p(5)*p(37),map_wdt*size_mini);
    D_p = reshape(sum(reshape(D_p',size_mini*map_wdt*p(1)*p(5),p(37)),2),map_wdt*size_mini,p(1)*p(5))'; 
    
    %Mask Up
    mask_2_2 = reshape(permute(reshape(mask_2_2',map_wdt,p(1)*p(5),size_mini),[2,1,3]),p(1)*p(5),map_wdt*size_mini);
    
    %Non-linearity second layer
    D_p(mask_2_2) = 0;
    if p(8)
        D_p = D_p.*(m_2>0);
    else
        D_p = D_p.*(1-m_2.^2);
    end
    
    %reshape for consistency with input to second layer
    D_p = reshape(D_p,p(1)*p(5)*map_wdt,size_mini);
else
    CR_3_b_df = 0; %third layer is unused
    CR_3_df = 0;
end

if p(10) >= 2 %if at least two layers
    
    D_p = reshape(D_p, p(1)*p(5), map_wdt*size_mini);
    if p(42) %dropout backprop at second layer
        D_p = D_p.*DRO_2;
    end
    CR_2_b_df = sum(D_p,2);
    
    %Down
    D_p = reshape(permute(reshape(D_p,p(1)*p(5),map_wdt,size_mini),[1,3,2]),p(1)*p(5)*size_mini,map_wdt);
   
    %% Backprop K-Max Pooling
    if p(31) %if GPU
        D_p = gather(D_p);
        subs_2 = gather(subs_2);
        q = round(gather(p));
        D_p_K = zeros(q(1)*q(5)*size_mini,q(26)+1,'single');
        D_p_k = zeros(size(D_p_K),'single');
        
        D_p_k(:,1:map_wdt) = D_p; %extend error with trailing zeros
        D_p_K(subs_2) = D_p_k;
        D_p_K = gpuArray(D_p_K(:,1:end-1));
    else
        D_p_K = zeros(p(1)*p(5)*size_mini,p(26)+1);
        D_p_k = zeros(size(D_p_K));
        
        D_p_k(:,1:map_wdt) = D_p; %extend error with trailing zeros
        D_p_K(subs_2) = D_p_k;
        D_p_K = D_p_K(:,1:end-1);
    end
    map_wdt = p(26); 
    
    
    
    if p(13) %Backprop folding
        %up
        D_p_K = reshape(permute(reshape(D_p_K',map_wdt,p(1),p(5)*size_mini),[2,1,3]),p(1),[]);
        
        D_p_K = repmat(D_p_K,2,1);
        p(1) = p(1)*2;
        %down
        D_p_K = reshape(permute(reshape(D_p_K,p(1),[],p(5)*size_mini),[1,3,2]),p(5)*p(1)*size_mini,[]);
    end
    
    
    %up
    D_p_K = repmat(reshape(permute(reshape(D_p_K',map_wdt,p(1)*p(5),size_mini),[2,1,3]),p(1),map_wdt*p(5)*size_mini),p(3),1);
    %down
    D_p_K = reshape(permute(reshape(D_p_K,p(1)*p(3)*p(5),map_wdt,size_mini),[1,3,2]),p(1)*p(3)*p(5)*size_mini,map_wdt);
    
    %compute and up
    f = reshape(permute(reshape(fastConv(fliplr(D_p_K),M_1,'v',p(31))',p(6),p(1)*p(3)*p(5),size_mini),[2,1,3]),p(1)*p(3)*p(5),p(6)*size_mini);
    CR_2_df = reshape(sum(reshape(f,p(1)*p(3)*p(5)*p(6),size_mini),2),p(1)*p(3)*p(5),p(6));

    D_p = fastConv(D_p_K, kernel_two, 'v',p(31)); 
    map_wdt = size(D_p,2);
    
    %Up and sum
    D_p = reshape(permute(reshape(D_p',map_wdt,p(1)*p(3)*p(5),size_mini),[2,1,3]),p(1)*p(3)*p(5),map_wdt*size_mini);
    D_p = reshape(sum(reshape(D_p',size_mini*map_wdt*p(1)*p(3),p(5)),2),map_wdt*size_mini,p(1)*p(3))'; 
    
    %Mask Up
    mask_1_2 = reshape(permute(reshape(mask_1_2',map_wdt,p(1)*p(3),size_mini),[2,1,3]),p(1)*p(3),map_wdt*size_mini);
    
    %Non-linearity first layer
    D_p(mask_1_2) = 0;
    if p(8)
        D_p = D_p.*(m_1>0);
    else
        D_p = D_p.*(1-m_1.^2);
    end
    
    %reshape for consistency with input to first layer
    D_p = reshape(D_p,p(1)*p(3)*map_wdt,size_mini);
    
else
    CR_2_b_df = 0; %third layer is unused
    CR_2_df = 0;
end

%up
D_p = reshape(D_p, p(1)*p(3), map_wdt*size_mini);
if p(41) %dropout at first layer
    D_p = D_p.*DRO_1;
end
CR_1_b_df = sum(D_p,2);

%Down
D_p = reshape(permute(reshape(D_p,p(1)*p(3),map_wdt,size_mini),[1,3,2]),p(1)*p(3)*size_mini,map_wdt);

%% Backprop K-Max Pooling
if p(31) %if GPU
    D_p = gather(D_p);
    subs_1 = gather(subs_1);
    q = round(gather(p));
    D_p_K = zeros(q(1)*q(3)*size_mini,q(25)+1,'single');
    D_p_k = zeros(size(D_p_K),'single');
    
    D_p_k(:,1:map_wdt) = D_p; %extend error with trailing zeros
    D_p_K(subs_1) = D_p_k;
    D_p_K = gpuArray(D_p_K(:,1:end-1));
else
    D_p_K = zeros(p(1)*p(3)*size_mini,p(25)+1);
    D_p_k = zeros(size(D_p_K));
    
    D_p_k(:,1:map_wdt) = D_p; %extend error with trailing zeros
    D_p_K(subs_1) = D_p_k;
    D_p_K = D_p_K(:,1:end-1);
end
map_wdt = p(25);

if p(12) %Backprop folding
    %up
    D_p_K = reshape(permute(reshape(D_p_K',map_wdt,p(1),p(3)*size_mini),[2,1,3]),p(1),[]);
    
    D_p_K = repmat(D_p_K,2,1);
    p(1) = p(1)*2;
    %down
    D_p_K = reshape(permute(reshape(D_p_K,p(1),[],p(3)*size_mini),[1,3,2]),p(3)*p(1)*size_mini,[]);
end

f = reshape(permute(reshape(fastConv(fliplr(D_p_K),data,'v',p(31))',p(4),p(1)*p(3),size_mini),[2,1,3]),p(1)*p(3),p(4)*size_mini);
CR_1_df = reshape(sum(reshape(f,p(1)*p(3)*p(4),size_mini),2),p(1)*p(3),p(4));
%% 
if p(32) 
    D_p = fastConv(D_p_K, kernel_one, 'v', p(31)); %error at data
    %%Backprop word embeddings
    %Up
    D_p = reshape(permute(reshape(D_p',p(2),p(1)*p(3),size_mini),[2,1,3]),p(1)*p(3),p(2)*size_mini);
    %Sum
    D_p = reshape(sum(reshape(D_p',p(1)*p(2)*size_mini,p(3)),2),p(2)*size_mini,p(1)); %No transpose because of next operation
    if p(40) %dropout at projection layer
       D_p = D_p.*DRO_0'; %transposed
    end
    % 
    if p(31)
        D_p = gather(D_p);
        minibatch = gather(minibatch);
        q = round(gather(p));
        minibatch = [minibatch,q(30)]; %Making sure error matrix size is right
        D_p = [D_p;zeros(1,q(1))];
        E_df = gpuArray(accumarray([repmat(minibatch',q(1),1),indices],D_p(:))');
    else
        %
        minibatch = [minibatch,p(30)]; %Making sure error matrix size is right
        D_p = [D_p;zeros(1,p(1))]; 
        E_df = accumarray([repmat(minibatch',p(1),1),indices],D_p(:))';
    end
else
    if p(31)
        E_df = gpuArray.zeros(size(CR_E),'single');
    else
        E_df = zeros(size(CR_E));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cost = -(1/size_mini)*cost;

if p(32), E_df = -(1/size_mini)*E_df; end
CR_1_df = -(1/size_mini)*CR_1_df;
CR_1_b_df = -(1/size_mini)*CR_1_b_df;
CR_2_df = -(1/size_mini)*CR_2_df;
CR_2_b_df = -(1/size_mini)*CR_2_b_df;
CR_3_df = -(1/size_mini)*CR_3_df;
CR_3_b_df = -(1/size_mini)*CR_3_b_df;
Z_df = -(1/size_mini)*Z_df;
if factor_b<0.00001
    CR_1_b_df = 0*CR_1_b_df;
    CR_2_b_df = 0*CR_2_b_df;
    CR_3_b_df = 0*CR_3_b_df;
end

% % %% Add regularization to gradient
CR_Z_nobias = CR_Z;
CR_Z_nobias(:,end) = zeros(size(CR_Z_nobias(:,end))); %OPTIM

if p(32), E_df = E_df + p(20) * CR_E; end;
CR_1_df = CR_1_df + p(21) * CR_1;
CR_2_df = CR_2_df + p(22) * CR_2;
CR_3_df = CR_3_df + p(23) * CR_3;
Z_df = Z_df + p(24) * CR_Z_nobias;
%% Add regularization to cost

if p(32) %if learning word embeddings
    cost = cost + ...
        (p(20)/2) * sum(sum(CR_E.^2)) + ...
        (p(21)/2) * sum(sum(CR_1.^2)) + ...
        (p(22)/2) * sum(sum(CR_2.^2)) + ...
        (p(23)/2) * sum(sum(CR_3.^2)) + ...
        (p(24)/2) * sum(sum(CR_Z(:,1:end-1).^2));
else
     cost = cost + ...
        (p(21)/2) * sum(sum(CR_1.^2)) + ...
        (p(22)/2) * sum(sum(CR_2.^2)) + ...
        (p(23)/2) * sum(sum(CR_3.^2)) + ...
        (p(24)/2) * sum(sum(CR_Z(:,1:end-1).^2)); %OPTIM
end
%% Return grad
[grad, dummy] = param2stack(E_df, CR_1_df, CR_1_b_df, CR_2_df, CR_2_b_df, CR_3_df, CR_3_b_df, Z_df, p(31));
return
