Self-Taught Convolutional Neural Networks for Short Text Clustering  
============================================================================  
@article{xu2017self,    
  title={Self-Taught Convolutional Neural Networks for Short Text Clustering},    
  author={Xu, Jiaming and Xu, Bo and Wang, Peng and Zheng, Suncong and Tian, Guanhua and Zhao, Jun and Xu, Bo},    
  journal={arXiv preprint arXiv:1701.00185},    
  year={2017}    
}    

**Note that:**  
----------------------------------------------------------------------------  
  
Here are instructions of the demo dataset&software for the paper [Self-Taught Convolutional Neural Networks for Short Text Clustering]    
  
**Usage:**  
    1. Please download the software and dataset packages, and put them into one folder;  
    2. The main function: ./software/main_STC2.m, please first "cd ./software/" and then run main_STC2.m via matlab;  
    
**Notices:**  
    1. The suggested memory of machine is 16GB RAM;  
    2. The suggested matlab version is R2011 and above;  
    3. This is a demo package which includes the all details about porposed method and baselines;  
    4. K-means clustering is very slow on original high-dimensionality (2W~3W dim.) text features;    
       If you want to run clustering via Kmeans, please have a little patience, and we strongly suggest that you directly refer the KMeans results in our paper which reports the average results by running KMeans 500 times;  
    5. Please feel free to send me emails if you have any problems in using this package.  

**Instructions of Archives:**  
    ./README.md: Some notices and instructions.  
    ./dataset/  
        -- Biomedical.txt: the raw 20,000 short text;  
        -- Biomedical_gnd.txt: the labels;  
        -- Biomedical_vocab2idx.dic: vocabulary index;  
        -- Biomedical_index.txt: has transfered the words into idx;  
        -- Biomedical-lite.mat: mini dataset only including feature vectors (fea) and labels (gnd);  
        -- Biomedical-STC2.mat: dataset for STC^2, including 20,000 short texts, 20 topics/tags and the pre-trained word embeddings;   
        -- SearchSnippets.txt: the raw 12,340 short text;  
        -- SearchSnippets_vocab2idx.dic: vocabulary index;  
        -- SearchSnippets_index.txt: has transfered the words into idx;  
        -- SearchSnippets-lite.mat: mini dataset only including feature vectors (fea) and labels (gnd);  
        -- SearchSnippets-STC2.mat: dataset for STC^2, including 12,340 short texts, 8 topics/tags and the pre-trained word embeddings;   
        -- StackOverflow.txt: the raw 20,000 short text;  
        -- StackOverflow_gnd.txt: the labels;  
        -- StackOverflow_vocab2idx.dic: vocabulary index;  
        -- StackOverflow_index.txt: has transfered the words into idx;  
        -- StackOverflow-lite.mat: mini dataset only including feature vectors (fea) and labels (gnd);          
        -- StackOverflow-STC2.mat: dataset for STC^2, including 20,000 short texts, 20 topics/tags and the pre-trained word embeddings;           
    ./software/: Main folder of software;  
				-- To be appeared soon.    