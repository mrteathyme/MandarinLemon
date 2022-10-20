from pyexpat import model
import tensorflow as tf

def createModel(sampleRate, audioBufferLength, architecture):
    return tf.keras.Sequential(architecture)

def buildModel(inputShape, outputs, outputActivator, hiddenLayers = []):
    model = [tf.keras.layers.InputLayer(input_shape = inputShape)]
    for i in range(len(hiddenLayers)):
        model.append(tf.keras.layers.Dense(hiddenLayers[i][0], activation=hiddenLayers[i][1]))
    model.append(tf.keras.layers.Dense(outputs, activation=outputActivator))
    return model
    #tf.keras.layers.Dense(4,activation='relu', input_shape=(None, int(sampleRate*(audioBufferLength/1000))+1)),





'''
    [
        tf.keras.layers.Dense(4,activation='relu', input_shape=(None, int(sampleRate*(audioBufferLength/1000))+1)),
        tf.keras.layers.Dense(108, activation='sigmoid')
    ]
'''