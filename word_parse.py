import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import gensim.models.keyedvectors as word2vec

class WordParse:
    def __init__(self, TRAIN_DATA_FILE, TEST_DATA_FILE, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
        # Regex to remove all Non-Alpha Numeric and space 将无用字符滤除例如“-”，“，”，“；”等等
        self.special_character_removal = re.compile(r'[^a-z\d ]', re.IGNORECASE)

        # regex to replace all numerics 将数字替换成n，数字并不对最终结果形成影响
        self.replace_numbers = re.compile(r'\d+', re.IGNORECASE)

        self.TRAIN_DATA_FILE = TRAIN_DATA_FILE
        self.TEST_DATA_FILE = TEST_DATA_FILE
        self.MAX_NB_WORDS = MAX_NB_WORDS
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.EMBEDDING_DIM = EMBEDDING_DIM

        self.list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

        return

    def get_train_test(self):
        train_df = pd.read_csv(self.TRAIN_DATA_FILE)
        test_df = pd.read_csv(self.TEST_DATA_FILE)

        list_sentences_train = train_df["comment_text"].fillna("NA").values
        list_sentences_test = test_df["comment_text"].fillna("NA").values

        train_y = train_df[self.list_classes].values

        print('Processing Training Data...')
        comments_train = self.__get_word_list(list_sentences_train)

        print('Processing Testing Data...')
        comments_test = self.__get_word_list(list_sentences_test)

        train_data, test_data, self.word_index = self.__get_word_keys(comments_train, comments_test)

        return train_data, test_data, train_y

    def get_embedding_matrix(self, all_embeddings_index, all_embeddings_dim, all_embedding_type):

        print('Preparing embedding matrix')
        nb_words = min(self.MAX_NB_WORDS, len(self.word_index))
        embedding_matrix = np.zeros((nb_words, self.EMBEDDING_DIM))
        dim_begin = 0
        for emb_idx, embeddings_index in enumerate(all_embeddings_index):
            embedding_dim = dim_begin + all_embeddings_dim[emb_idx]
            print(embedding_dim)
            for word, i in self.word_index.items():
                if i >= self.MAX_NB_WORDS:
                    continue
                if all_embedding_type[emb_idx] == 'word2vec':
                    embedding_vector = embeddings_index.get(word)
                else:
                    embedding_vector = embeddings_index.get(str.encode(word, 'utf-8'))
                #
                # print(embedding_vector)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i, dim_begin:embedding_dim] = embedding_vector

            dim_begin = embedding_dim

        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

        return embedding_matrix

    def __get_word_keys(self, train_comments, test_comments):
        tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS)
        tokenizer.fit_on_texts(train_comments + test_comments)

        sequences = tokenizer.texts_to_sequences(train_comments)
        test_sequences = tokenizer.texts_to_sequences(test_comments)

        word_index = tokenizer.word_index
        print('Found %s unique tokens' % len(word_index))

        train_data = pad_sequences(sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        print('Shape of data tensor:', train_data.shape)

        test_data = pad_sequences(test_sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        print('Shape of test_data tensor:', test_data.shape)

        return train_data, test_data, word_index

    def __get_word_list(self, sentences):
        comments = []
        for ii, text in enumerate(sentences):
            if ii % 100000 == 0:
                print(str(ii) + ' samples...')
            comments.append(self.__text_to_wordlist(text))
        return comments

    def word2vec(self, EMBEDDING_FILE, EMBEDDING_TYPE):
        print('Indexing word vectors')
        # word vectors
        embeddings_index = {}
        if EMBEDDING_TYPE == 'glove' or EMBEDDING_TYPE == 'fasttext':
            f = open(EMBEDDING_FILE, 'rb')
            for ii, line in enumerate(f):
                if ii % 100000 == 0:
                    print(str(ii) + ' words ...')
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()
        elif EMBEDDING_TYPE == 'word2vec':
            word2vecDict = word2vec.KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
            for ii, word in enumerate(word2vecDict.vocab):
                if ii % 100000 == 0:
                    print(str(ii) + ' words ...')
                embeddings_index[word] = word2vecDict.word_vec(word)
        else:
            print('No such kind of word vectors!')
        print('Total %s word vectors.' % len(embeddings_index))

        return embeddings_index

    def __text_to_wordlist(self, text, remove_stopwords=False, stem_words=False):
        # Clean the text, with the option to remove stopwords and to stem words.

        # Convert words to lower case and split them
        text = text.lower().split()

        # Optionally, remove stop words
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]

        text = " ".join(text)

        # Remove Special Characters
        text = self.special_character_removal.sub('', text)

        # Replace Numbers
        text = self.replace_numbers.sub('n', text)

        # Optionally, shorten words to their stems
        if stem_words:
            text = text.split()
            stemmer = SnowballStemmer('english')
            stemmed_words = [stemmer.stem(word) for word in text]
            text = " ".join(stemmed_words)

        # Return a list of words
        return (text)
