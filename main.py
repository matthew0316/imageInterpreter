import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import random
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# loading data set ( 70k images --> 28x28 --> 10 different images )
fashion_mnist = keras.datasets.fashion_mnist

# pull out the train and test data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# neural net structure
model = keras.Sequential([
    # input layer
    # the input is a (28px x 28px) image which is Flattened into a (784px x 1px) image one pixel per node or 784 neurons
    keras.layers.Flatten(input_shape=(28, 28)),


    # hidden layer
    # there is not fixed or mathematical way to figure out units, just 'play around' and see what number works best
    # Dense means that every node is connected to every node in every column
    keras.layers.Dense(units=128, activation=tf.nn.relu),

    # output layer
    # units == nodes, ie 10 images for 10 nodes
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

# compiling the model
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy')

# train the model
model.fit(train_images, train_labels, epochs=5)

# testing the model
test_loss = model.evaluate(test_images, test_labels)

num = random.randint(0, 10000)

# make predictions
predictions = model.predict(test_images)

# refine predictions
specificPredictions = predictions[num]

max = 0
maxIndex = 0

for i in range(len(specificPredictions)):
    if specificPredictions[i] > max:
        max = specificPredictions[i]
        maxIndex = i

print("\nThe Correct Answer: {}".format(test_labels[num]))
print("List Of Predictions: {}".format(predictions[num]))
print("The Computer's Guess: {}".format(maxIndex))

if test_labels[num] == 0:
    print('The Image: A Top')
    if maxIndex == 0:
        print('The CPU: Guessed Correctly')

elif test_labels[num] == 1:
    print('The Image: Pair of Trousers')
    if maxIndex == 1:
        print('The CPU: Guessed Correctly')

elif test_labels[num] == 2:
    print('The Image: A Pullover')
    if maxIndex == 2:
        print('The CPU: Guessed Correctly')

elif test_labels[num] == 3:
    print('The Image: A Dress')
    if maxIndex == 3:
        print('The CPU: Guessed Correctly')

elif test_labels[num] == 4:
    print('The Image: A Coat')
    if maxIndex == 4:
        print('The CPU: Guessed Correctly')

elif test_labels[num] == 5:
    print('The Image: A Sandal')
    if maxIndex == 5:
        print('The CPU: Guessed Correctly')

elif test_labels[num] == 6:
    print('The Image: A Shirt')
    if maxIndex == 6:
        print('The CPU: Guessed Correctly')

elif test_labels[num] == 7:
    print('The Image: A Sneaker')
    if maxIndex == 7:
        print('The CPU: Guessed Correctly')

elif test_labels[num] == 8:
    print('The Image: A Bag')
    if maxIndex == 8:
        print('The CPU: Guessed Correctly')

elif test_labels[num] == 9:
    print('The Image: An Ankle Boot')
    if maxIndex == 9:
        print('The CPU: Guessed Correctly')

plt.imshow(test_images[num], cmap='gray')
plt.show()
plt.close()