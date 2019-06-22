import keras
from keras import layers, models
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np


def quantize_arr(arr):
    """Quantization based on linear rescaling over min/max range.
    """
    min_val, max_val = np.min(arr), np.max(arr)
    if max_val - min_val > 0:
        quantized = np.round(255 * (arr - min_val) / (max_val - min_val))
    else:
        quantized = np.zeros(arr.shape)
    quantized = quantized.astype(np.uint8)
    min_val = min_val.astype(np.float32)
    max_val = max_val.astype(np.float32)
    return quantized, min_val, max_val


def LeNet():

    #Instantiate an empty model
    model = Sequential()

    # C1 Convolutional Layer
    model.add(
        layers.Conv2D(6,
                      kernel_size=(5, 5),
                      strides=(1, 1),
                      activation="tanh",
                      input_shape=(28, 28, 1),
                      padding="same"))

    # S2 Pooling Layer
    model.add(
        layers.AveragePooling2D(pool_size=(2, 2),
                                strides=(1, 1),
                                padding="valid"))

    # C3 Convolutional Layer
    model.add(
        layers.Conv2D(16,
                      kernel_size=(5, 5),
                      strides=(1, 1),
                      activation="tanh",
                      padding="valid"))

    # S4 Pooling Layer
    model.add(
        layers.AveragePooling2D(pool_size=(2, 2),
                                strides=(2, 2),
                                padding="valid"))

    # C5 Fully Connected Convolutional Layer
    model.add(
        layers.Conv2D(120,
                      kernel_size=(5, 5),
                      strides=(1, 1),
                      activation="tanh",
                      padding="valid"))
    #Flatten the CNN output so that we can connect it with fully connected layers
    model.add(layers.Flatten())

    # FC6 Fully Connected Layer
    model.add(layers.Dense(84, activation="tanh"))

    #Output Layer with softmax activation
    model.add(layers.Dense(10, activation="softmax"))

    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer="SGD",
                  metrics=["accuracy"])

    return model


def Train(model, output_path, x_train, y_train):

    model.fit(x=x_train,
              y=y_train,
              epochs=10,
              batch_size=128,
              validation_data=(x_test, y_test),
              verbose=1)
    model.save(output_path)


def Test(model, weights_path, x_test, y_test):

    # Load Weights
    model.load_weights(weights_path)

    test_score = model.evaluate(x_test, y_test)
    print("Test loss {:.4f}, accuracy {:.2f}%".format(test_score[0],
                                                      test_score[1] * 100))


def print_weights(model, weights_path):

    # Load Weights
    model.load_weights(weights_path)

    for layer in model.layers:
        print(layer.get_weights())


def write_file(output_path, weights):

    with open(output_path, 'w') as f:
        f.write(str(weights))


def Quantize_weights(weights_float):

    weights_uint8 = []
    layers = []

    for weight_float in weights_float:
        for a in weight_float:
            weight, min_val, max_val = quantize_arr(a)
            weights_uint8.append(weight)
    layers.append(weights)

    return layers


if __name__ == "__main__":

    # Load dataset as train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # # Set numeric type to float32 from uint8
    # x_train = x_train.astype('uint8')
    # x_test = x_test.astype('uint8')

    # Normalize value to [0, 1]
    # x_train /= 255
    # x_test /= 255

    # Transform lables to one-hot encoding
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # Reshape the dataset into 4D array
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Model initilization
    model = LeNet()

    # MNIST data fit (Comment this line if you already made the training)
    Train(model, "LeNet_5.h5", x_train, y_train)

    # Test accuracy
    Test(model, "LeNet_5.h5", x_test, y_test)

    # print float32/64 weights
    print_weights(model, "LeNet_5.h5")