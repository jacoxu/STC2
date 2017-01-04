# -*- encoding:utf-8 -*-
"""
    TEST Skip-thought vector 2016/10/26
"""
import skipthoughts
import numpy

dataset = 'Biomedical'  # 1. SearchSnippets 2. StackOverflow 3. Biomedical
print 'Step 1: start to load model ...'
model = skipthoughts.load_model()
print 'Step 2: start to load data ...'
data_path = './../../dataset/' + dataset + '.txt'
X = []
for _, line in enumerate(open(data_path)):
    X.append(line)

print 'Step 3: start to encode the data to vectors ...'
vectors = skipthoughts.encode(model, X)
print 'Step 4: save the vectors to file ...'
vectors_file = './results/' + dataset + '.vec'
numpy.savetxt(vectors_file, vectors, delimiter=', ', fmt='%1.5e')
print len(vectors), ', ', len(vectors[0]), ', ', vectors[0]
print 'It\'s done'
