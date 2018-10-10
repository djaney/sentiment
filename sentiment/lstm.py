import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical


def build_model(data, num_classes=None, epoch=1, verbose=1):
    x = []
    y = []

    for d in data:
        x.append(d[0])
        y.append(d[1])

    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(x)
    x = tokenizer.texts_to_sequences(x)
    x = pad_sequences(x)
    y = to_categorical(y, num_classes=num_classes)

    embed_dim = 128
    lstm_out = 196

    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length=x.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    batch_size = 32
    model.fit(x, y, epochs=epoch, batch_size=batch_size, verbose=verbose)

    return tokenizer, model, x.shape[1]


def predict(tokenizer_model_maxlen, data):
    tokenizer, model, maxlen = tokenizer_model_maxlen
    x = tokenizer.texts_to_sequences([data])

    x = pad_sequences(x, maxlen=maxlen)
    predictions = model.predict(x)
    prediction_class = np.argmax(predictions[0])
    return prediction_class
