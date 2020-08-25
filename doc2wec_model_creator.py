import pandas as pd
import logging
import random
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

version = 18

model_name = "models/teached_d2w_v{}".format(version)

data = pd.read_csv("data/cleaned_textes_v{}.csv".format(version))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def read_corpus(data, tokens_only=False):
    for i, line in enumerate(data):
        tokens = simple_preprocess(line)
        if tokens_only:
            yield tokens
        else:
            # For training data, add tags
            yield TaggedDocument(tokens, [i])

print()
train_corpus = list(read_corpus(data.text[:len(data.text)-(len(data.text) - 1919)]))
test_corpus = list(read_corpus(data.text[len(data.text)-(len(data.text) - 1919):], tokens_only=True))
print("corpus lenght:", train_corpus)

#create model
model = Doc2Vec(vector_size=60, min_count=2, epochs=40)

#build vocab
model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
model.save(model_name)

# Pick a random document from the test corpus and infer a vector from the model
"""
doc_id = random.randint(0, len(test_corpus) - 1)
inferred_vector = model.infer_vector(test_corpus[doc_id])
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))"""
