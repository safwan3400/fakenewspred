from nltk.stem.porter import PorterStemmer
import numpy as np  # linear algebra
import pandas as pd  # data processing

import tensorflow as tf

import os
import re
import nltk


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

from nltk.corpus import stopwords
nltk.download('stopwords')




def get_all_query(news_title, author_name, text):
    total = news_title+' '+author_name+' '+text
    return total


def remove_punctuation_stopwords_lemma(total):
  ps = PorterStemmer()

  review = re.sub('[^a-zA-Z]', ' ', total)
  review= review.lower()
  review= review.split()
  review= [ps.stem(word) for word in review if not word in stopwords.words('english')]
  review= ' '.join(review)

  corpus_test = []
  corpus_test.append(review)
  voc_size = 5000
  onehot_rep_test = [one_hot(words,voc_size)for words in corpus_test]
  embedded_docs_test = pad_sequences(onehot_rep_test, padding='pre', maxlen=25)

  test_final = np.array(embedded_docs_test)

  return test_final
