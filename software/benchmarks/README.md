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
  
Here are instructions of the classification baselines for the three short text datasets.  
  
**Usage:**  
>1. The main function: ./software/benchmarks/Classification_ACC.m;  
    
**Notices:**  
>1. The classification methods are SVM-liner and SVM-RBF;  
>2. The features are TF, TFIDF and AE (average embedding);  
>3. Please feel free to send me emails if you have any problems in using this package.  

**Instructions of Archives:**  
>./README.md: Some notices and instructions.  
>./Classification_ACC.m: Test the classification performance with SVM, and the results are listed below.  
>./normalize.m: normalize the feature vectors;  
>./predict.mexw64: LibSVM libraries;  
>./svmpredict.mexw64  
>./svmtrain.mexw64  
>./train.mexw64  
>./tf_idf.m: Compute TF-IDF;  

**ACC Results:**  

Classification Methods|SearchSnippets|StackOverflow|Biomedical
 ------------- |:-------------:|:-------------:|:-------------:
SVM-Linear (TF)|	67.72|	83.70|	71.48
SVM-Linear (TFIDF)|	70.96|	84.55|	71.55
SVM-Kernel (TF)|	62.32|	79.05|	68.73
SVM-Kernel (TFIDF)|	64.78|	82.23	|70.85
SVM-Linear (AE)|	87.15|	81.90	|62.75
SVM-Kernel (AE)|	87.63	|81.43|	62.80
