import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gzip import GzipFile

from word2gauss import GaussianEmbedding, iter_pairs
from word2gauss.words import Vocabulary

# load the vocabulary
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

# create the embedding to train
# use 100 dimensional spherical Gaussian with KL-divergence as energy function
embed = GaussianEmbedding(vocab._ntokens, 100,
    covariance_type='spherical', energy_type='KL')

# open the corpus and train with 8 threads
# the corpus is just an iterator of documents, here a new line separated
# gzip file for example
with open('../multisense-prob-fasttext/data/text8', 'r') as corpus:
    embed.train(iter_pairs(corpus, vocab), n_workers=8)

# save the model for later
embed.save('modelfiles/test', vocab=vocab.id2word, full=True)