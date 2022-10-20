import tensorflow as tf
import tensorflow_addons as tfa
import random
import numpy as np
import model as mod
import notation

ATuning = 440.0
Tuning = 440.0 # will be user input later
sampleRate = 44100
audioBufferLength = 100
numInputNodes = 4411
trainingDataSamples = 100000
trainingEpochs = 50
outputs = 108
trainingData, expectedOutput = notation.generateTrainingData(trainingDataSamples, sampleRate, audioBufferLength, 6)
print(expectedOutput.shape)
print(trainingData.shape)
print(int(trainingDataSamples/(trainingEpochs/5)))

def randomiseLayers(minLayers, maxLayers, minNodes, maxNodes, activatorFunction, inputs, outputs, nodeBudget=0):
    layers = []
    maxNodes = min(maxNodes, inputs)
    if nodeBudget > 0:
        maxNodes = min(maxNodes, nodeBudget)
        budget = True
    else:
        budget = False
    for i in range(random.randrange(minLayers,maxLayers+1)):
        nodeCount = min(max(min(random.randrange(minNodes,maxNodes+1),maxNodes),outputs),nodeBudget)
        if nodeCount < outputs:
            return layers
        layers.append([nodeCount, activatorFunction])
        maxNodes = max(nodeCount,outputs)
        if budget == True:
            if nodeBudget <= 0 or nodeBudget < outputs:
                return layers
            else:
                nodeBudget -= nodeCount
    return layers

alpha = 2
nodeBudget = trainingDataSamples / (alpha * ((int(sampleRate*(audioBufferLength/1000))+1)+outputs))
#layerStructure = randomiseLayers(1, 10, 2, 2000, 'relu', int(sampleRate*(audioBufferLength/1000))+1, outputs, 2000)
layerStructure = [[300, 'relu'], [200, 'relu']]
arch = mod.buildModel((None, int(sampleRate*(audioBufferLength/1000))+1),outputs,'sigmoid',layerStructure)
model = mod.createModel(sampleRate, audioBufferLength, arch)

print(layerStructure)

#model.load_weights('model.h5')
metric = tfa.metrics.HammingLoss(mode='multilabel', threshold=0.8)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy',metric],
)


my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='hamming_loss',patience=10),
    #tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]


def train(callbacks=[]):
    return model.fit(
    trainingData,
    expectedOutput,
    epochs=trainingEpochs,
    batch_size=int(trainingDataSamples/2),
    callbacks = callbacks
    )
results = [0,0]
bestResult = 0
bestLayer = []
attemptedLayers = []
attemptedLayers.append(layerStructure)

train(callbacks=my_callbacks)
model.save_weights('model.h5')



results = model.evaluate(trainingData, expectedOutput, batch_size=10)
#print(results)
#print(layerStructure)
#print(bestLayer, bestResult)
#print(expectedOutput[0])
#print(trainingData[0])
test = np.reshape(trainingData[3], (1,4411))
prediction = model.predict(test)
np.argwhere(prediction[0] > 0.5)
np.argwhere(expectedOutput[3] == 1)
#print(model.predict(test))
#print(model.predict(test)-expectedOutput[0])

trainingData[0].shape