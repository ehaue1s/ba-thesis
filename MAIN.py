#Evaluating Document Level Sentiment Analysis Models: CNN vs RNN
#Elias Haueis

import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras.layers import Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Activation
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
import matplotlib as plt 

# Hyperparamenters
dir_name = os.getcwd()
path_data_amazon = dir_name+"\Kindle.json" #path to data set, https://nijianmo.github.io/amazon/index.html,
label_x = "reviewText"
label_y = "overall"
NUM_REVIEWS_TRAIN = 2000000
NUM_REVIEWS_VAL = 4000000
NUM_REVIEWS_TEST = 4100000
MAX_LEN = 300           #maximale Länge der Docs
BATCH_SIZE = 128
EPOCHS = 7
SHUFFLE = True
NUM_FILTERS = 1000
KERNEL_SIZE = 100

#Data Preprocessing
data = pd.read_json(path_data_amazon) #Data in JSON 
data = data.fillna(value="None")
data_train = data[0:NUM_REVIEWS_TRAIN]
data_val = data[NUM_REVIEWS_TRAIN:NUM_REVIEWS_VAL]
data_test = data[NUM_REVIEWS_VAL:NUM_REVIEWS_TEST]
train_x = data_train[label_x].values
val_x = data_val[label_x].values
train_y = data_train[label_y].values
val_y = data_val[label_y].values
test_x = data_test[label_x].values 
test_y_RAW = data_test[label_y].values
train_y = keras.utils.to_categorical(train_y, 6) 
val_y = keras.utils.to_categorical(val_y, 6)
test_y = keras.utils.to_categorical(test_y_RAW, 6)
train_y = np.array(train_y)
val_y = np.array(val_y)
test_y = np.array(test_y)
list_texts = []
for i in train_x:
    list_texts.append(i)

#Word Embeddings
Tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token="oovtoken")
Tokenizer.fit_on_texts(list_texts)
VOCAB_SIZE = len(Tokenizer.word_index) + 1
train_seq = Tokenizer.texts_to_sequences(train_x)
train_seq = pad_sequences(train_seq, maxlen=MAX_LEN)
val_seq = Tokenizer.texts_to_sequences(val_x)
val_seq = pad_sequences(val_seq, maxlen=MAX_LEN)
test_seq = Tokenizer.texts_to_sequences(test_x)
test_seq = pad_sequences(test_seq, maxlen=MAX_LEN)

#CNN Model
CNN = Sequential()
CNN.add(Embedding(input_dim=VOCAB_SIZE, output_dim=32, input_length=MAX_LEN))
CNN.add(Conv1D(filters=NUM_FILTERS, kernel_size=KERNEL_SIZE,  activation="relu"))
CNN.add(GlobalMaxPooling1D())
CNN.add(Dense(6, activation="softmax"))
CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
CNN.summary()

#RNN Model
RNN = Sequential()
RNN.add(Embedding(input_dim=VOCAB_SIZE, output_dim=32, input_length=MAX_LEN))
RNN.add(LSTM(1024, dropout=0.01))
RNN.add(Dense(6, activation="softmax"))
RNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
RNN.summary()
#Hybrid Model
HYBRID = Sequential()
HYBRID.add(Embedding(input_dim=VOCAB_SIZE, output_dim=32, input_length=MAX_LEN))
HYBRID.add(Conv1D(filters= NUM_FILTERS, kernel_size= KERNEL_SIZE, activation="relu"))
HYBRID.add(GlobalMaxPooling1D(keepdims=True))
HYBRID.add(LSTM(1024, dropout=0.01))
HYBRID.add(Dense(6, activation="softmax"))
HYBRID.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
HYBRID.summary()

#Model Training
CNN.fit(train_seq, train_y, validation_data=(val_seq,val_y), batch_size=BATCH_SIZE, shuffle=SHUFFLE, epochs=EPOCHS)
RNN.fit(train_seq, train_y, validation_data=(val_seq, val_y), batch_size=BATCH_SIZE, shuffle=SHUFFLE, epochs=EPOCHS)
HYBRID.fit(train_seq, train_y, validation_data=(val_seq, val_y), batch_size=BATCH_SIZE, shuffle=SHUFFLE, epochs=EPOCHS)


#Model Evaluation
score_CNN = CNN.evaluate(x=test_seq, y=test_y)
score_RNN = RNN.evaluate(x=test_seq, y=test_y)
score_HYBRID = HYBRID.evaluate(x=test_seq, y=test_y)
def print_score(score):
    print('Test loss:', score[0]) 
    print('Test accuracy:', score[1])
print_score(score_CNN)
print_score(score_RNN)
print_score(score_HYBRID)


#Testing
def get_predictions(model, test_sequence):
    predictions = model.predict(test_sequence)
    pred = []
    for i in predictions:
        pred.append(np.argmax(i))
    return pred

pred_cnn = get_predictions(CNN, test_seq)
pred_rnn = get_predictions(RNN, test_seq)
pred_hybrid = get_predictions(HYBRID, test_seq)



#Confusion Matrix
def build_cm(pred, test_raw, name):
    cm = confusion_matrix(test_raw, pred, labels=[0,1,2,3,4,5])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2,3,4,5])
    disp.plot()
    plt.pyplot.savefig(dir_name+ name)


build_cm(pred_cnn, test_y_RAW,"confusioncnn.jpg")
build_cm(pred_rnn, test_y_RAW, "confusionrnn.jpg")
build_cm(pred_hybrid, test_y_RAW, "confusionhybrid.jpg")


#saving Models

def save_model(model, modelname_string):
    model.save(dir_name + modelname_string +".h5")
    print("Saved model"+modelname_string)

save_model(CNN, "CNN")
save_model(RNN, "RNN")
save_model(HYBRID, "Hybrid")

#classification report
print("CNN classification report:")
print(classification_report(pred_cnn, test_y_RAW))
print("RNN classification report:")
print(classification_report(pred_rnn, test_y_RAW))
print("HYBRID classification report:")
print(classification_report(pred_hybrid, test_y_RAW))