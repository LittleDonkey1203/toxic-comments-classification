from word_parse import WordParse
from model import nlp_model
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import random
import numpy as np

def make_submission(SUBMISSION_FILE, y_test, submisstion_tag):
    # Make submission file & log file to tracing back
    sample_submission = pd.read_csv(SUBMISSION_FILE)
    sample_submission[word_parse.list_classes] = y_test
    sample_submission.to_csv(submission_path + str(EMBEDDING_TYPE) + '_' + nlp_model.model_tag + '_' + submisstion_tag +  '.csv', index=False)

if __name__ == '__main__':
    data_path = 'data/'
    embd_path = 'embeds/'
    model_path = 'saved_model/'
    submission_path = 'submission_file/'
    loc_path = 'log/'

    NO_BALANCE = True

    TRAIN_DATA_FILE = data_path + 'train.csv'
    TEST_DATA_FILE = data_path + 'test.csv'
    SUBMISSION_FILE = submission_path + 'sample_submission.csv'

    EMBEDDING_FILE_glove = embd_path + 'glove.840B.300d.txt'     # Standford GloVe word2vec database
    EMBEDDING_FILE_fasttext = embd_path + 'crawl-300d-2M.vec'     # Facebook Fasttext word2vec database
    EMBEDDING_FILE_word2vec = embd_path + 'GoogleNews-vectors-negative300.bin'  # Google word2vec database

    EMBEDDING_TYPE_glove = 'glove'
    EMBEDDING_TYPE_fasttext = 'fasttext'
    EMBEDDING_TYPE_word2vec = 'word2vec'

    EMBEDDING_TYPE = [EMBEDDING_TYPE_fasttext]

    EMBEDDING_DIM_glove = 300  # feature number of word vector
    EMBEDDING_DIM_fasttext = 300
    EMBEDDING_DIM_word2vec = 300

    EMBEDDING_DIM = [EMBEDDING_DIM_fasttext]

    MAX_SEQUENCE_LENGTH = 150     # max sequence of each sentence
    MAX_NB_WORDS = 2000000         # max word quoted in GloVe

    VALIDATION_SPLIT = 0.1
    BATCH_SIZE_TRAIN = 256
    BATCH_SIZE_TEST = 256
    EPOCHS = 50
    OUTPUT_SIZE = 6
    BAGGING_K = 5

    OUTPUT_FEATURE_NUM = 300  # LSTM output features
    DENSE_HIDDEN_NUM = 256               # Dense hidden layer neurals
    DROP_OUT_RATE_LSTM = 0.25         # Dropout parameters
    DROP_OUT_RATE_DENSE = 0.25        # Dropout parameters

    word_parse = WordParse(TRAIN_DATA_FILE, TEST_DATA_FILE, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, sum(EMBEDDING_DIM))
    # embeddings_index_glove = word_parse.word2vec(EMBEDDING_FILE_glove, EMBEDDING_TYPE_glove)   # Get GloVe word vectors Index
    embeddings_index_fasttext = word_parse.word2vec(EMBEDDING_FILE_fasttext, EMBEDDING_TYPE_fasttext)   # Get Fasttext word vectors Index
    # embeddings_index_word2vec = word_parse.word2vec(EMBEDDING_FILE_word2vec, EMBEDDING_TYPE_word2vec)   # Get Word2vec word vectors Index

    embedding_index = [embeddings_index_fasttext]

    list_sentences_train, list_sentences_test, y_train =\
        word_parse.get_train_test()    # Get train and test input and train target


    embedding_matrix = word_parse.get_embedding_matrix(
        embedding_index,
        EMBEDDING_DIM,
        EMBEDDING_TYPE)  # Get embedding matrix by embedding index, this step has to be after encoded the all scentences, and has word_list information

    # del embeddings_index_glove
    del embeddings_index_fasttext
    # del embeddings_index_word2vec

    print(min(MAX_NB_WORDS, len(word_parse.word_index)))

    nlp_model = \
        nlp_model(BATCH_SIZE_TRAIN, BATCH_SIZE_TEST, EPOCHS,
                  sum(EMBEDDING_DIM), OUTPUT_SIZE, OUTPUT_FEATURE_NUM, DENSE_HIDDEN_NUM, MAX_SEQUENCE_LENGTH,
                  DROP_OUT_RATE_LSTM, DROP_OUT_RATE_DENSE,
                  min(MAX_NB_WORDS, len(word_parse.word_index)), model_path)   # Initialized basic parameters

    kf = StratifiedKFold(n_splits=BAGGING_K, shuffle=True)
    bagging_iter = 0
    y_test = 0
    for train_idx, val_idx in kf.split(list_sentences_train, np.any(y_train == 1, axis = 1)):
        print('BAGGING:' + str(bagging_iter) + ' PREDICTION,,,')
        data_tr = list_sentences_train[train_idx]
        y_tr = y_train[train_idx]
        data_vl = list_sentences_train[val_idx]
        y_vl = y_train[val_idx]

        ind = np.any(y_tr == 1, axis = 1)
        data_tr_positive = data_tr[ind,:]
        y_tr_positive = y_tr[ind,:]
        data_tr_negtive = data_tr[~ind,:]
        y_tr_negtive = y_tr[~ind,:]

        kf_sub = KFold(n_splits=10, shuffle=True)

        sub_bagging_iter = 0
        y_test_sub = 0
        for idx1, idx in kf_sub.split(y_tr_negtive):
            data_subset_tr = np.concatenate((data_tr_positive, data_tr_negtive[idx]), axis = 0)
            y_subset_tr = np.concatenate((y_tr_positive, y_tr_negtive[idx]), axis = 0)
            print('data structure:' + str(np.sum(y_subset_tr == 1, axis = 0) / len(y_subset_tr)))
            nlp_model.get_bidirectional_lstm_att(embedding_matrix, str(EMBEDDING_TYPE) + '_' + str(bagging_iter) + '_' + str(sub_bagging_iter))   # Get lstm attention model
            nlp_model.fit(data_subset_tr, y_subset_tr, data_vl, y_vl)    # Fit the model

            nlp_model.load_model()
            print(nlp_model.bst_model_path + ' model loaded')

            if bagging_iter == 0:
                y_test_sub = nlp_model.predict(list_sentences_test)   # Get predictions
            else:
                y_test_sub += nlp_model.predict(list_sentences_test)
            sub_bagging_iter += 1
        y_test_sub = y_test_sub / 10
        make_submission(SUBMISSION_FILE, y_test_sub, 'iter_' + str(bagging_iter))

        if bagging_iter == 0:
            y_test = y_test_sub
        else:
            y_test += y_test_sub
        bagging_iter += 1

    y_test = y_test / BAGGING_K
    make_submission(SUBMISSION_FILE, y_test, 'iter_total')