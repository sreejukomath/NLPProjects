# create sample documents
doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health."

#compile sample documents
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

# Get stop words
from stop_words import get_stop_words

#create english sop words
en_stop = get_stop_words('en')

# init stemmers
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

tokenizer = RegexpTokenizer(r'\w+')
p_stemmer = PorterStemmer()

texts = []

# loop through document list
for i in doc_set:

    # clean and tokenize
    raw = i.tower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    # add tokens to list
    texts.append(stemmed_tokens)

from gensim import corpora, models

import genism

# turn our tokenized document into term dictionary
dictionary = corpora.dictionary(texts)

# convert tokenized document into into document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passess=20)

print(ldamodel.print_topics(num_topics=3, mum_words=3))