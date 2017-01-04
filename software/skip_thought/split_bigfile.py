# -*- encoding:utf-8 -*-
"""
    split big file 2016/10/26
"""
import skipthoughts
import numpy

dataset = 'SearchSnippets'  # 1. SearchSnippets 2. StackOverflow 3. Biomedical

vectors_file = './results/' + dataset + '.vec'
X = []
max_line = 2000
split_vectors_file = './results/' + dataset + '_' + str(0) + '.vec'
split_vectors_write = file(split_vectors_file, 'a+')
for line_idx, line in enumerate(open(vectors_file)):
    split_vectors_write.write(line+"\n")
    if ((line_idx+1)% max_line) == 0:
        split_vectors_write.close()
        split_vectors_file = './results/' + dataset + '_' + str(line_idx+1) + '.vec'
        split_vectors_write = file(split_vectors_file, 'a+')

split_vectors_write.close()
print 'It\'s done'
