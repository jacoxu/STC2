function C = fastConv(A,B,type,gpu)

% perform "matrixized" fast convolution of matrix A and B along
% dimension dim. The convolution is fast because it uses the matrix
% operation (and fft/ifft) only. 
%
% for example:
%
%      fastConv(A,B) performs convolution of corresponding
%      rows in A,B
%
% currently dim = 2 is fixed (only row-wise)
%
% the code is equivalent to running conv(A_i,B_i, 'full') in matlab
% (where A_i and B_i are columns (dim=1) or rows (dim=2) of A,B)
% and then stack the results together
%

%% Padding
% a1 = size(A,1); %Must be same as b2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% gradient check
% if type == 'f'
%     C = zeros(size(A,1),size(A,2)+size(B,2)-1);
% else
%     C = zeros(size(A,1),size(A,2)-size(B,2)+1);
% end
% 
% for i=1:size(A,1)
%     C(i,:) = conv(A(i,:),B(i,:),type);
% end
% 
% return;

a2 = size(A,2);
b1 = size(B,1);
b2 = size(B,2);

if gpu
    if a2>=b2
        p = (a2-b2)/2;
        B = [gpuArray.zeros(b1,floor(p),'single'),B,gpuArray.zeros(b1,ceil(p),'single')];
    else
        p = (b2-a2)/2;
        A = [gpuArray.zeros(b1,floor(p),'single'),A,gpuArray.zeros(b1,ceil(p),'single')];
    end

    %% Calculation

    A = [A,gpuArray.zeros(size(A),'single')];
    B = [B,gpuArray.zeros(size(B),'single')];

    C = real(ifft(fft(A,[],2).*fft(B,[],2),[],2));

else
    if a2>=b2
        p = (a2-b2)/2;
        B = [zeros(b1,floor(p)),B,zeros(b1,ceil(p))];
    else
        p = (b2-a2)/2;
        A = [zeros(b1,floor(p)),A,zeros(b1,ceil(p))];
    end
    
    %% Calculation
    
    A = [A,zeros(size(A))];
    B = [B,zeros(size(B))];
    
    C = ifft(fft(A,[],2).*fft(B,[],2),[],2);
end

%% Trimming

if type=='f'
    if a2>=b2
        p = (a2-b2)/2;
        C = C(:,floor(p)+1:end-1-ceil(p));
    else
        p = (b2-a2)/2;
        C = C(:,floor(p)+1:end-1-ceil(p));
    end
else
    if a2>=b2
        p = (a2+b2-2)/2;
        C = C(:,floor(p)+1:end-1-ceil(p));
    else
        C = [];
    end
end








