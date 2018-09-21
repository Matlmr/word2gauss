from word2gauss import GaussianEmbedding
from word2gauss.words import Vocabulary

# load in a previously trained model and the vocab
with open('../multisense-prob-fasttext/modelfiles/text8_singlvar_dim50.words', 'r') as document:
    answer = {}
    i = 0
    for line in document:
        line = line.split()
        if not line:  # empty line?
            continue
        answer[line[0]] = i
        i = i + 1
vocab = Vocabulary(answer)
embed = GaussianEmbedding.load('modelfiles/test')

# find nearest neighbors to 'rock'
print 'Words and their similarity'
for i in embed.nearest_neighbors('brother', vocab=vocab):
    print i['word'], i['similarity']

# find nearest neighbors to 'rock' sorted by covariance
print '\n Words and their covariance'
for i in embed.nearest_neighbors('brother', num=10, vocab=vocab, sort_order='sigma'):
    print i['word'], i['sigma']

# solve king + woman - man = ??
print'Word vectors algebra'
for i in embed.nearest_neighbors([['king', 'woman'], ['man']], num=10, vocab=vocab):
    print i['word']
