%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We modified the code from
% A convolutional neural network for modelling sentences
% Blunsom, Phil, Edward Grefenstette, and Nal Kalchbrenner
% Conference on the 52nd Annual Meeting of the Association for Computational Linguistics (ACL 2014)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function STC2_CNN(dataset,parameters,evaluteScore,randinit4Methods,randinit4KMeans,repeatNum,nbcluster)

p(1) = parameters.wordDim;  disp(strcat('Size word vectors:',num2str(p(1))));

%% for word embedding
p(32) = 200;                disp(strcat('Word embedding learning ON:',num2str(p(32))));
p(33) = 199;                disp(strcat('if emb learn ON, after how many epochs OFF:',num2str(p(33))));
p(34) = 1;                  disp(strcat('use preinitialized vocabulary:',num2str(p(34))));

%% Iter number
p(61) = 200; disp(strcat('Batch size:',num2str(p(61))));
p(62) = parameters.iter; disp(strcat('maxEpochs:',num2str(p(62))));
p(63) = 0; 
p(64) = 0.01;disp(strcat('Learning rate:',num2str(p(64))));%gamma = 0.01;
p(72) = 1;
p(65) = 0;disp(strcat('Bias_b:',num2str(p(65))));

% max is p(78) now

%% Start to loop for all structure of NN
for p8=0 %Relu
p(8)=p8; disp(strcat('Using relu:',num2str(p(8))));
for p40=0; %Dropout on word embedding layer
p(40)=p40; disp(strcat('Dropout at Projection Sentence matrix:',num2str(p(40))));
for p7=5; %k_top
p(7)=p7;  disp(strcat('TOP POOLING width:',num2str(p(7))));  %k_top
for p10=2; %Layer of CNN
p(10)=p10; disp(strcat('Number of conv layers being used (1 or 2):',num2str(p(10))));
%% the first layer
for p3=12;
p(3)=p3; disp(strcat('Number feat maps in first layer:',num2str(p(3)))); 
for p4=3;
p(4)=p4; disp(strcat('Size of kernel in first layer:', num2str(p(4))));
for p41=0
p(41)=p41; disp(strcat('Dropout at First layer:',num2str(p(41)))); 
for p12=1 
p(12)=p12;  disp(strcat('Folding in first layer:', num2str(p(12))));
    %% the second layer
    for p5=8;
    p(5)=p5; disp(strcat('Number feat maps in second layer:', num2str(p(5))));
    for p6=3
    p(6)=p6; disp(strcat('Size of kernel in second layer:', num2str(p(6))));
    for p42=1;
    p(42)=p42;  disp(strcat('Dropout at Second layer:',num2str(p(42))));
    for p13=1;
        p(13)=p13;  disp(strcat('Folding in second layer:', num2str(p(13))));
        if p(10)==2 
            for p78=1;
                p(78)=p78;
                tic;
                filePrefixStr = ([num2str(p(8)),'-',num2str(p(7)),'-',num2str(p(3)),'-',num2str(p(5)),'-',...
                num2str(p(42)),'-',num2str(p(13)),'-',num2str(p(78)*1000),'-STCC']);
                disp(['filePrefixStr:',filePrefixStr])
                Train(dataset,p,filePrefixStr,evaluteScore,randinit4Methods,randinit4KMeans,repeatNum,nbcluster,parameters);
                toc;
            end
        else
            error(['Input the worry layer number:',p(10)]);
        end
    end
    end
    end
    end
end
end
end
end
end
end
end
end
end