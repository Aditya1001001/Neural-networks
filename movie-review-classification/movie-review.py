import tensorflow as tf
from tensorflow import keras
import numpy as np
from random import randint


dataset = keras.datasets.imdb
#only take words with more than a certain amount of total occurences
(X_train, Y_train), (X_test, Y_test) = dataset.load_data(num_words=10000)

#what the data looks like
#print(X_train[0])

# A dictionary mapping words to an integer index, courtesy of tensorflow
_word_index = dataset.get_word_index()
word_index = {k:(v+3) for k,v in _word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2         #unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])   

def decode_review(text):
    '''returns the decoded (human readable) reviews'''  
    return " ".join([reverse_word_index.get(i, "?") for i in text]) #if we don't get an index we put a ?

#decoded data
#print(decode_review(X_train[0]))

#reviews have different amount of words
#print(len(X_train[0]), len(X_train[100]),sep="  ")

#making every review of lenght 300
X_train = keras.preprocessing.sequence.pad_sequences(X_train, value=word_index["<PAD>"], padding="post", maxlen=300)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, value=word_index["<PAD>"], padding="post", maxlen=300)
# print(len(X_train[0]), len(X_train[100]),sep="  ")

# model = keras.Sequential()
# # A word embedding layer will attempt to determine the meaning of each word in the sentence by mapping each word to a position in vector space. 
# # It will bring word vectors that are used in similar context together in the vector space, think of it as just grouping similar words together.
# model.add(keras.layers.Embedding(88000, 16))
# # GlobalAveragePooling1D scales down the 16 dimensions. The 1D Global average pooling block takes a 2-dimensional tensor  of size (input size) x (input channels) 
# # and computes the maximum of all the (input size) values for each of the (input channels).
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16, activation="relu"))
# model.add(keras.layers.Dense(1, activation="sigmoid"))

# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# # prints a summary of the model
# model.summary()  

# # spliiting the training set to get validation data 
# X_val = X_train[:10000]
# X_train = X_train[10000:]

# Y_val = Y_train[:10000]
# Y_train = Y_train[10000:]

# # batch_size is to accomodate for the lack of memory on system, some dataset might be too big to load and train at once.
# # verbose is a flag that controls what we see while our model is training
# fit_model = model.fit(X_train, Y_train, epochs=40, batch_size=512, validation_data=(X_val, Y_val), verbose=1)

# accuracy = model.evaluate(X_test, Y_test)
# print(accuracy)

# model.save("model.h5")

model = keras.models.load_model("model.h5")

# for _ in range(5):
#     i = randint(0,500)
#     test_review = X_test[i]
#     prediction = model.predict(test_review)
#     print("Review:  ")
#     print(decode_review(test_review))
#     print("Prediction-  ", str(prediction[0]))
#     print("Actual-  ", str(Y_test[i]))

def review_encode(s):
	encoded = [1]

	for word in s:
		if word.lower() in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)
	return encoded

#   testing on outside data
with open("test_review.txt", encoding="UTF-8") as f:
    for line in f.readlines():
        #getting rid of unecessary characters
        nline = line.replace(",","").replace(".","").replace("(","").replace(")","").replace(":","").replace("-","").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        # make the data 300 words long
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=300)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])
