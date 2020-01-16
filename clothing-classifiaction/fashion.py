import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt
from random import randint

dataset = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#visualizing the data
# plt.imshow(train_images[0])
# plt.savefig("boots.png")
# plt.show()
# print(train_images[0])

#scaling the data
train_images = train_images/255.0
test_images = test_images/255.0

#model creation: We will do this by using the Sequential object from keras.
#input layers flattens the data
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax")
	])

#setting parameters for the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#training the model
#epoch = "how many times does the neural network sees the data"
model.fit(train_images, train_labels, epochs=7)

#evaluating the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(test_accuracy)

#making predictions 
predictions = model.predict(test_images)

#visualizing five random predictions
plt.figure(figsize=(5,5))
for _ in range(5):
    i = randint(0,5000)
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual label:   " + class_names[test_labels[i]])
    plt.title("Predicted label:    " + class_names[np.argmax(predictions[i])])
    plt.show()