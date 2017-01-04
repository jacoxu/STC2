#!/bin/bash
# SearchSnippets, StackOverflow, Biomedical
dataset="SearchSnippets"
rm alldata-id.txt
rm vectors.txt
if [ "$dataset" == "SearchSnippets" ]
then
	echo "Current dataset is SearchSnippets ..."
	awk 'BEGIN{a=0;}{print "_*" a " " $0; a++;}' < SearchSnippets_index.txt > alldata-id.txt
elif [ "$dataset" == "StackOverflow" ]
then
	echo "Current dataset is StackOverflow ..."
	awk 'BEGIN{a=0;}{print "_*" a " " $0; a++;}' < StackOverflow_index.txt > alldata-id.txt
elif [ "$dataset" == "Biomedical" ]
then
	echo "Current dataset is Biomedical ..."
	awk 'BEGIN{a=0;}{print "_*" a " " $0; a++;}' < Biomedical_index.txt > alldata-id.txt
else
	echo "Current dataset is wrong!"${dataset}
fi
gcc word2vec.c -o word2vec -lm -pthread -O3 -march=native -funroll-loops
time ./word2vec -train ./alldata-id.txt -output vectors.txt -cbow 0 -size 100 -window 10 -negative 5 -hs 0 -sample 1e-4 -threads 40 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1
if [ "$dataset" == "SearchSnippets" ]
then
	grep '_\*' vectors.txt > SearchSnippets_para2vecs.txt
elif [ "$dataset" == "StackOverflow" ]
then
	grep '_\*' vectors.txt > StackOverflow_para2vecs.txt
elif [ "$dataset" == "Biomedical" ]
then
	grep '_\*' vectors.txt > Biomedical_para2vecs.txt
else
	echo "Current dataset is wrong!"${dataset}
fi
echo "It is done, OK!"