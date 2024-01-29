from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def define_dense_model_with_hidden_layers(input_length, activation_func_array=['sigmoid','sigmoid'],
                                          hidden_layers_sizes=[50, 20], output_function='softmax', output_length=10):
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(input_length,)))
    for i, size in enumerate(hidden_layers_sizes):
        model.add(layers.Dense(size, activation=activation_func_array[i]))
    model.add(layers.Dense(output_length, activation=output_function))
    return model


def set_layers_to_trainable(model, trainable_layer_numbers):
    for i, layer in enumerate(model.layers):
        layer.trainable = i in trainable_layer_numbers
    return model
