from keras import backend as K
import numpy as np
# from keras import initializations
from keras import initializers, regularizers, constraints
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.engine.topology import Layer
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Bidirectional, Conv1D, MaxPool1D, Flatten, GRU, Concatenate, SpatialDropout1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
config.gpu_options.per_process_gpu_memory_fraction = 0.95
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim

class multi_nlp_model():
    def __init__(self, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST, EPOCHS,
                 EMBEDDING_DIM, OUTPUT_SIZE, OUTPUT_FEATURE_NUM, DENSE_HIDDEN_NUM, MAX_SEQ_LEN,
                 DROP_OUT_RATE_LSTM, DROP_OUT_RATE_DENSE,
                 MAX_NB_WORDS,
                 MODEL_PATH):
        self.batch_size = BATCH_SIZE_TRAIN
        self.batch_size_test = BATCH_SIZE_TEST
        self.epochs = EPOCHS
        self.embedding_dim = EMBEDDING_DIM
        self.output_size = OUTPUT_SIZE
        self.output_feature_num = OUTPUT_FEATURE_NUM
        self.dense_hidden_num = DENSE_HIDDEN_NUM
        self.max_seq_len = MAX_SEQ_LEN
        self.dropout_rate_lstm = DROP_OUT_RATE_LSTM
        self.dropout_rate_dense = DROP_OUT_RATE_DENSE
        self.word_index_num = MAX_NB_WORDS
        self.model_path = MODEL_PATH

    def get_bidirectional_multi_lstm_att(self, embedding_matrix, model_idx):
        comment_input = Input(shape=(self.max_seq_len,))
        lstm_out = []
        for emb_mat_tmp in embedding_matrix:
            embedding_layer = (Embedding(self.word_index_num, self.embedding_dim,
                                    weights=[emb_mat_tmp],
                                    input_length=self.max_seq_len, trainable=False))
            lstm_layer = (Bidirectional(LSTM(self.output_feature_num,
                          dropout=self.dropout_rate_lstm,
                          recurrent_dropout=self.dropout_rate_lstm,
                          return_sequences=True)))
            embedded_sequences = embedding_layer(comment_input)
            lstm_out.append(lstm_layer(embedded_sequences))
        x = 0
        for ii, out_tmp in enumerate(lstm_out):
            if ii == 0:
                x = out_tmp
            else:
                x = Concatenate()([x, out_tmp])
        x = Dropout(self.dropout_rate_dense)(x)
        merged = Attention(self.max_seq_len)(x)
        merged = Dense(self.dense_hidden_num, activation='relu')(merged)
        merged = Dropout(self.dropout_rate_dense)(merged)
        merged = BatchNormalization()(merged)
        y = Dense(self.output_size, activation='sigmoid')(merged)

        model = Model(inputs=[comment_input], outputs=y)
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        print(model.summary())

        model_tag = 'bidirectional_multi_lstm_attention_glove_vectors_drop_params_%.2f_%.2f' % (
        self.dropout_rate_lstm, self.dropout_rate_dense)
        bst_model_path = self.model_path + model_tag + '_model_idx_' + model_idx + '.h5'
        print(model_tag)
        self.reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        self.model = model
        self.model_tag = model_tag

    def fit(self, data_train, labels_train, data_val, labels_val):
        print('training data size: ' + str(data_train.shape))
        print('training target size: ' + str(labels_train.shape))
        print('validation data size: ' + str(data_val.shape))
        print('validation target size: ' + str(labels_val.shape))
        self.hist = self.model.fit(data_train, labels_train,
                                   verbose=2,
                                   validation_data=(data_val, labels_val),
                                   epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                                   callbacks=[self.early_stopping, self.model_checkpoint, self.reduce_lr_on_plateau])

    def predict(self, data):
        print('testing data size: ' + str(data.shape))
        y = self.model.predict([data], batch_size=self.batch_size_test, verbose=2)
        return y

    def predict_from_saved_model(self, data, saved_model_path):
        y = 0
        print('testing data size: ' + str(data.shape))
        model_lists = os.listdir(saved_model_path)
        for ii, model_list in enumerate(model_lists):
            print('iteration ' + str(ii))
            model_tmp = load_model(saved_model_path + model_list)
            if ii == 0:
                y = model_tmp.predict([data], batch_size=self.batch_size_test, verbose=2)
            else:
                y = y + model_tmp.predict([data], batch_size=self.batch_size_test, verbose=2)
        return y

class nlp_model():
    def __init__(self, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST, EPOCHS,
                 EMBEDDING_DIM, OUTPUT_SIZE, OUTPUT_FEATURE_NUM, DENSE_HIDDEN_NUM, MAX_SEQ_LEN,
                 DROP_OUT_RATE_LSTM, DROP_OUT_RATE_DENSE,
                 MAX_NB_WORDS,
                 MODEL_PATH):
        self.batch_size = BATCH_SIZE_TRAIN
        self.batch_size_test = BATCH_SIZE_TEST
        self.epochs = EPOCHS
        self.embedding_dim = EMBEDDING_DIM
        self.output_size = OUTPUT_SIZE
        self.output_feature_num = OUTPUT_FEATURE_NUM
        self.dense_hidden_num = DENSE_HIDDEN_NUM
        self.max_seq_len = MAX_SEQ_LEN
        self.dropout_rate_lstm = DROP_OUT_RATE_LSTM
        self.dropout_rate_dense = DROP_OUT_RATE_DENSE
        self.word_index_num = MAX_NB_WORDS
        self.model_path = MODEL_PATH

    def get_bidirectional_gru_att(self, embedding_matrix, model_idx):
        embedding_layer = Embedding(self.word_index_num, self.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_seq_len, trainable=False)
        gru_layer = Bidirectional(GRU(self.output_feature_num,
                          dropout=self.dropout_rate_lstm,
                          recurrent_dropout=self.dropout_rate_lstm,
                          return_sequences=True))
        comment_input = Input(shape=(self.max_seq_len,))
        embedded_sequences = embedding_layer(comment_input)
        x = gru_layer(embedded_sequences)
        x = Dropout(self.dropout_rate_dense)(x)
        merged = Attention(self.max_seq_len)(x)
        merged = Dense(self.dense_hidden_num, activation='relu')(merged)
        merged = Dropout(self.dropout_rate_dense)(merged)
        merged = BatchNormalization()(merged)
        y = Dense(self.output_size, activation='sigmoid')(merged)

        model = Model(inputs=[comment_input], outputs=y)
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        print(model.summary())

        model_tag = 'bidirectional_gru_attention_glove_vectors_drop_params_%.2f_%.2f' % (
        self.dropout_rate_lstm, self.dropout_rate_dense)
        bst_model_path = self.model_path + model_tag + '_model_idx_' + str(model_idx) + '.h5'
        print(model_tag)
        self.reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        self.model = model
        self.model_tag = model_tag

    def get_bidirectional_lstm_conv1d(self, embedding_matrix, model_idx):
        embedding_layer = Embedding(self.word_index_num, self.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_seq_len, trainable=False)
        lstm_layer = Bidirectional(LSTM(self.output_feature_num,
                                        dropout=self.dropout_rate_lstm,
                                        recurrent_dropout=self.dropout_rate_lstm,
                                        return_sequences=True))
        comment_input = Input(shape=(self.max_seq_len,))
        embedded_sequences = embedding_layer(comment_input)
        x = lstm_layer(embedded_sequences)
        x = Dropout(self.dropout_rate_dense)(x)
        feat_num = self.output_feature_num
        conv1 = Conv1D(feat_num, 3, padding='same')(x)
        conv1 = Conv1D(feat_num, 3, padding='same')(conv1)
        maxpl1 = MaxPool1D(padding='same')(conv1)
        norm1 = BatchNormalization()(maxpl1)
        conv2 = Conv1D(feat_num, 3, padding='same')(norm1)
        conv2 = Conv1D(feat_num, 3, padding='same')(conv2)
        maxpl2 = MaxPool1D(padding='same')(conv2)
        norm2 = BatchNormalization()(maxpl2)
        conv3 = Conv1D(feat_num, 3, padding='same')(norm2)
        conv3 = Conv1D(feat_num, 3, padding='same')(conv3)
        maxpl3 = MaxPool1D(padding='same')(conv3)
        norm3 = BatchNormalization()(maxpl3)
        conv4 = Conv1D(feat_num, 3, padding='same')(norm3)
        conv4 = Conv1D(feat_num, 3, padding='same')(conv4)
        maxpl4 = MaxPool1D(padding='same')(conv4)
        norm4 = BatchNormalization()(maxpl4)
        conv5 = Conv1D(feat_num, 3, padding='same')(norm4)
        conv5 = Conv1D(feat_num, 3, padding='same')(conv5)
        maxpl5 = MaxPool1D(padding='same')(conv5)
        norm5 = BatchNormalization()(maxpl5)

        merged = Flatten()(norm5)
        merged = Dropout(self.dropout_rate_dense)(merged)
        merged = Dense(self.dense_hidden_num, activation='relu')(merged)
        merged = Dropout(self.dropout_rate_dense)(merged)
        merged = BatchNormalization()(merged)

        y = Dense(self.output_size, activation='sigmoid')(merged)

        model = Model(inputs=[comment_input], outputs=y)
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        print(model.summary())

        model_tag = 'bidirectional_lstm_conv1d_glove_vectors_drop_params_%.2f_%.2f' % (
            self.dropout_rate_lstm, self.dropout_rate_dense)
        bst_model_path = self.model_path + model_tag + '_model_idx_' + str(model_idx) + '.h5'
        print(model_tag)
        self.reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        self.model = model
        self.model_tag = model_tag


    def get_conv1d(self, embedding_matrix, model_idx):
        embedding_layer = Embedding(self.word_index_num, self.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_seq_len, trainable=False)
        comment_input = Input(shape=(self.max_seq_len,))
        embedded_sequences = embedding_layer(comment_input)
        feat_num = self.output_feature_num
        conv1 = Conv1D(feat_num, 3, padding='same')(embedded_sequences)
        conv1 = Conv1D(feat_num, 3, padding='same')(conv1)
        maxpl1 = MaxPool1D(padding='same')(conv1)
        norm1 = BatchNormalization()(maxpl1)
        conv2 = Conv1D(feat_num, 3, padding='same')(norm1)
        conv2 = Conv1D(feat_num, 3, padding='same')(conv2)
        maxpl2 = MaxPool1D(padding='same')(conv2)
        norm2 = BatchNormalization()(maxpl2)
        conv3 = Conv1D(feat_num, 3, padding='same')(norm2)
        conv3 = Conv1D(feat_num, 3, padding='same')(conv3)
        maxpl3 = MaxPool1D(padding='same')(conv3)
        norm3 = BatchNormalization()(maxpl3)
        conv4 = Conv1D(feat_num, 3, padding='same')(norm3)
        conv4 = Conv1D(feat_num, 3, padding='same')(conv4)
        maxpl4 = MaxPool1D(padding='same')(conv4)
        norm4 = BatchNormalization()(maxpl4)
        conv5 = Conv1D(feat_num, 3, padding='same')(norm4)
        conv5 = Conv1D(feat_num, 3, padding='same')(conv5)
        maxpl5 = MaxPool1D(padding='same')(conv5)
        norm5 = BatchNormalization()(maxpl5)

        merged = Flatten()(norm5)
        merged = Dropout(self.dropout_rate_dense)(merged)
        merged = Dense(self.dense_hidden_num, activation='relu')(merged)
        merged = Dropout(self.dropout_rate_dense)(merged)
        merged = BatchNormalization()(merged)

        y = Dense(self.output_size, activation='sigmoid')(merged)

        model = Model(inputs=[comment_input], outputs=y)
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        print(model.summary())

        model_tag = 'conv1d_glove_vectors_drop_params_%.2f_%.2f' % (
            self.dropout_rate_lstm, self.dropout_rate_dense)
        bst_model_path = self.model_path + model_tag + '_model_idx_' + str(model_idx) + '.h5'
        print(model_tag)
        self.reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        self.model = model
        self.model_tag = model_tag


    def get_bidirectional_lstm_att(self, embedding_matrix, model_idx):
        embedding_layer = Embedding(self.word_index_num, self.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_seq_len, trainable=False)
        lstm_layer = Bidirectional(LSTM(self.output_feature_num,
                          dropout=self.dropout_rate_lstm,
                          recurrent_dropout=self.dropout_rate_lstm,
                          return_sequences=True))
        comment_input = Input(shape=(self.max_seq_len,))
        embedded_sequences = embedding_layer(comment_input)
        x = lstm_layer(embedded_sequences)
        x = Dropout(self.dropout_rate_dense)(x)
        merged = Attention(self.max_seq_len)(x)
        merged = Dense(self.dense_hidden_num, activation='relu')(merged)
        merged = Dropout(self.dropout_rate_dense)(merged)
        merged = BatchNormalization()(merged)
        y = Dense(self.output_size, activation='sigmoid')(merged)

        model = Model(inputs=[comment_input], outputs=y)
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        print(model.summary())

        model_tag = 'bidirectional_lstm_attention_glove_vectors_drop_params_%.2f_%.2f' % (
        self.dropout_rate_lstm, self.dropout_rate_dense)
        bst_model_path = self.model_path + model_tag + '_model_idx_' + model_idx + '.h5'
        print(model_tag)
        self.reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        self.model = model
        self.model_tag = model_tag

    def get_model_lstm_att(self, embedding_matrix, model_idx):
        embedding_layer = Embedding(self.word_index_num, self.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_seq_len, trainable=False)
        lstm_layer = LSTM(self.output_feature_num,
                          dropout=self.dropout_rate_lstm,
                          recurrent_dropout=self.dropout_rate_lstm,
                          return_sequences=True)
        comment_input = Input(shape=(self.max_seq_len,))
        embedded_sequences = embedding_layer(comment_input)
        x = lstm_layer(embedded_sequences)
        x = Dropout(self.dropout_rate_dense)(x)
        merged = Attention(self.max_seq_len)(x)
        merged = Dense(self.dense_hidden_num, activation='relu')(merged)
        merged = Dropout(self.dropout_rate_dense)(merged)
        merged = BatchNormalization()(merged)
        y = Dense(self.output_size, activation='sigmoid')(merged)

        model = Model(inputs=[comment_input], outputs=y)
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        print(model.summary())

        model_tag = 'lstm_attention_glove_vectors_drop_params_%.2f_%.2f' % (self.dropout_rate_lstm, self.dropout_rate_dense)
        bst_model_path = self.model_path + model_tag + '_model_idx_' + str(model_idx) + '.h5'
        print(model_tag)
        self.reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        self.model = model
        self.model_tag = model_tag

    def fit(self, data_train, labels_train, data_val, labels_val):
        print('training data size: ' + str(data_train.shape))
        print('training target size: ' + str(labels_train.shape))
        print('validation data size: ' + str(data_val.shape))
        print('validation target size: ' + str(labels_val.shape))
        self.hist = self.model.fit(data_train, labels_train,
                       verbose=2,
                       validation_data=(data_val, labels_val),
                       epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                       callbacks=[self.early_stopping, self.model_checkpoint, self.reduce_lr_on_plateau])

    def predict(self, data):
        print('testing data size: ' + str(data.shape))
        y = self.model.predict([data], batch_size=self.batch_size_test, verbose=2)
        return y

    def predict_from_saved_model(self, data, saved_model_path):
        y = 0
        print('testing data size: ' + str(data.shape))
        model_lists = os.listdir(saved_model_path)
        print(str(len(model_lists)) + ' models loaded...')
        for ii, model_list in enumerate(model_lists):
            print('iteration '+str(ii))
            self.model.load_weights(saved_model_path+model_list)
            print('model '+model_list+' loaded')
            if ii == 0:
                y = self.model.predict([data], batch_size=self.batch_size_test, verbose=2)
            else:
                y = y + self.model.predict([data], batch_size=self.batch_size_test, verbose=2)
        y = y / len(model_lists)
        return y