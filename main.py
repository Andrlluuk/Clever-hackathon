import pandas as pd
import numpy as np
import keras
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dropout
from keras.preprocessing import *
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.preprocessing import LabelEncoder
import os


def load_data_from_arrays(strings, labels, train_test_split=0.9):
    data_size = len(strings)
    test_size = int(data_size - round(data_size * train_test_split))

    x_train = strings[test_size:]
    y_train = labels[test_size:]

    x_test = strings[:test_size]
    y_test = labels[:test_size]

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    PATH = str(os.getcwd())
    questions = pd.read_csv(PATH + '/data.csv', sep=";", index_col="ID")
    test = pd.read_csv(PATH + '/train.csv', sep=";", index_col="ID")
    t = []
    for input, label in zip(questions["Question"][:30000], test["Answer"]):
        t.append((word_tokenize(input), label))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list(questions['Question']))

    textSequences = tokenizer.texts_to_sequences(list(questions['Question']))

    X_train, y_train, X_test, y_test = load_data_from_arrays(textSequences, test["Answer"], train_test_split=0.8)
    max_words = 0
    for desc in questions["Question"].tolist():
        words = len(desc.split())
        if words > max_words:
            max_words = words
    total_unique_words = len(tokenizer.word_counts)

    maxSequenceLength = max_words

    vocab_size = total_unique_words + 1

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(questions["Question"])

    XX_train = sequence.pad_sequences(X_train, maxlen=maxSequenceLength)
    XX_test = sequence.pad_sequences(X_test, maxlen=maxSequenceLength)

    YY_train = keras.utils.to_categorical(y_train, 2)
    YY_test = keras.utils.to_categorical(y_test, 2)

    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)

    num_classes = 2

    max_features = vocab_size

    model = Sequential()
    model.add(Embedding(max_features, maxSequenceLength))
    model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    batch_size = 32
    epochs = 1

    history = model.fit(XX_train, YY_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(XX_test, YY_test))

    final_test = tokenizer.sequences_to_matrix(textSequences[30000:], mode='binary')

    test_sentences = textSequences[30000:]
    test_sentences = sequence.pad_sequences(test_sentences, maxlen=maxSequenceLength)

    score = model.evaluate(XX_test, YY_test,
                           batch_size=batch_size, verbose=1)

    score = model.predict(test_sentences)
    probs = score.transpose()[1]
    test = pd.read_csv(PATH + 'test.csv', sep=";", index_col="ID")
    test.insert(0, "Predict", probs)
    test.to_csv('results.csv', header=False)