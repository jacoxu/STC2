[Self-Taught Convolutional Neural Networks for Short Text Clustering](https://arxiv.org/abs/1701.00185)  
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
>1. Please download the software and dataset packages, and put them into one folder;  
>2. The main function: ./software/main_STC2.m, please first "cd ./software/" and then run main_STC2.m via matlab;  
    
**Notices:**  
>1. The suggested memory of machine is 16GB RAM;  
>2. The suggested matlab version is R2011 and above;  
>3. This is a demo package which includes the all details about porposed method and baselines;  
>4. K-means clustering is very slow on original high-dimensionality (2W~3W dim.) text features;    
>If you want to run clustering via Kmeans, please have a little patience, and we strongly suggest that you directly refer the KMeans results in our paper which reports the average results by running KMeans 500 times;  
>5. Please feel free to send me emails if you have any problems in using this package.  

**Instructions of Archives:**  
>./README.md: Some notices and instructions.  
>./dataset/  
>>-- Biomedical.txt: the raw 20,000 short text;  
>>-- Biomedical_gnd.txt: the labels;  
>>-- Biomedical_vocab2idx.dic: vocabulary index;  
>>-- Biomedical_index.txt: has transfered the words into idx;  
>>-- Biomedical-lite.mat: mini dataset only including feature vectors (fea) and labels (gnd);  
>>-- Biomedical-STC2.mat: dataset for STC^2, including 20,000 short texts, 20 topics/tags and the pre-trained word embeddings;   
>>-- SearchSnippets.txt: the raw 12,340 short text;  
>>-- SearchSnippets_vocab2idx.dic: vocabulary index;  
>>-- SearchSnippets_index.txt: has transfered the words into idx;  
>>-- SearchSnippets-lite.mat: mini dataset only including feature vectors (fea) and labels (gnd);  
>>-- SearchSnippets-STC2.mat: dataset for STC^2, including 12,340 short texts, 8 topics/tags and the pre-trained word embeddings;   
>>-- StackOverflow.txt: the raw 20,000 short text;  
>>-- StackOverflow_gnd.txt: the labels;  
>>-- StackOverflow_vocab2idx.dic: vocabulary index;  
>>-- StackOverflow_index.txt: has transfered the words into idx;  
>>-- StackOverflow-lite.mat: mini dataset only including feature vectors (fea) and labels (gnd);          
>>-- StackOverflow-STC2.mat: dataset for STC^2, including 20,000 short texts, 20 topics/tags and the pre-trained word embeddings;           

>./software/: Main folder of software;  
>>-- main_STC2.m: main function, and select one clustering method here: Kmeans, RecNN, AveEmbedding, LSA, Spectral_LE, etc.;  
>>-- run.sh: running it on commond line for linux user rather than window user;  
>>-- STC2.m: interfaces of clustering methods;  
>>-- STC2_CNN.m: interfaces of DCNN;  
>>-- AE/: Average Embedding (AE) folder;  
>>-- DCNN/: Dynamic Convolutional Neural Network (DCNN)[1] folder;  
>>-- LE/: Laplacian Eigenmaps (LE)[2] folder;  
>>-- LPI/: Locality Preserving Indexing (LPI)[3] folder;  
>>-- LSA/: Latent Semantic Analysis (LSA)[4] folder;  
>>-- Para2vec/: Paragraph vector (Para2vec)[5] folder;  
>>-- RecNN/: Recursive Neural Network (RecNN)[6] folder;  
>>-- results/: All evaluate results (ACC and NMI) of clustering will be saved in this folder;  
>>-- tools/: Tool folder;  
>>-- benchmarks/: Contains some classification benchmarks, SVM-linear or SVM-RBF on TF, TFIDF or AE. Get more classification details into this folder.  
				

**References:**  				
>[1]. N. Kalchbrenner, E. Grefenstette, P. Blunsom, A convolutional neural network for modelling sentences, ACL, 2014.  
>[2]. M. Belkin, P. Niyogi, Laplacian eigenmaps and spectral techniques for embedding and clustering, NIPS, 2001.  
>[3]. D. Cai, X. He, J. Han, Document clustering using locality preserving indexing, IEEE Transactions on Knowledge and Data Engineering, 2005.  
>[4]. S. C. Deerwester, S. T. Dumais, T. K. Landauer, G. W. Furnas, R. A. Harshman, Indexing by latent semantic analysis, JAsIs, 1990.  
>[5]. Q. Le, T. Mikolov, Distributed representations of sentences and documents, ICML, 2014.  
>[6]. R. Socher, J. Pennington, E. H. Huang, A. Y. Ng, C. D. Manning, Semisupervised recursive autoencoders for predicting sentiment distributions, EMNLP, 2011.  

