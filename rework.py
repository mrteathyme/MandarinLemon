import tensorflow as tf
import tensorflow_addons as tfa
import random
import numpy as np
import model as mod
import audioPreProcessor as ap
#import notation



def generateTrainingData(num, samples):
    data = []
    expectedOutput = []
    notes = []
    notes.append([2,'A',0])
    notes.append([2,'B',2])
    notes.append([3,'C',2])
    for i in range(num):
                #note, octave, frequency, interval = generateNote(Tuning)
        #data.append(np.insert(generateSamples(audioBufferLength/1000, frequency, sampleRate), 0, normalizedTuning))
        notesinner = []
        octaves = []
        intervals = []
        frequencies = []
        if random.choice([True,False]):
            chordNotes = random.randrange(3,maxChordNotes+1)-1
            for i in range(chordNotes):
                note, octave, frequency, interval = generateNote(Tuning)
                notesinner.append(note)
                octaves.append(octave)
                intervals.append(interval)
                frequencies.append(frequency)
            frequency = sum(frequencies)/chordNotes
        else:
            note, octave, frequency, interval = generateNote(Tuning)
            notesinner = [note]
            octaves = [octave]
            intervals = [interval]
            frequencies = [frequency]
        data.append(ap.chordBuilder(samples[440], notes))
        expectedOutputInner = []
        for y in range(108):
            if y in intervals:
                expectedOutputInner.append(1)
            else:
                expectedOutputInner.append(0)
        expectedOutput.append(np.array(expectedOutputInner))
    return (np.array(data), np.array(expectedOutput))




frequency = 440
samples = 44100
duration = 100

t = np.linspace(0, duration, samples)
np.sin(2*np.pi*frequency*t).shape



trainingData, frequenciesGen, notesGen, intervalsGen, octavesGen = notation.generateTestingData(440, 44100, 100, 6)
TestingData.shape

samples = ap.loadData()
samples[440][2]['A'][0].shape
notes = []
notes.append([2,'A',0])
ap.chordBuilder(samples[440], notes).shape

notes = []
notes.append([2,'A',0])
notes.append([2,'B',2])
notes.append([3,'C',2])
'''
trainingData = []
trainingData.append()
trainingData.append(ap.chordBuilder(samples[440], notes))
trainingData.append(ap.chordBuilder(samples[440], notes))
'''
expectedOutput = []
expectedOutput.append(np.zeros(108))
expectedOutput[0][36] = 1
expectedOutput[0][34] = 1
expectedOutput[0][35] = 1
expectedOutput = np.array(expectedOutput)
print(np.array(expectedOutput).shape)
'''
sampleRate = 44100
audioLength = 100
print(trainingData[0])
'''

model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape = (None, 4410)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(108, activation='sigmoid')
    ]
)
#print(layerStructure)

#model.load_weights('model.h5')


metric = tfa.metrics.HammingLoss(mode='multilabel', threshold=0.8)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy',metric],
)


my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='hamming_loss',patience=10),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

trainingData.shape
def train(callbacks=[]):
    return model.fit(
    trainingData,
    expectedOutput,
    epochs=20,
    batch_size=1,
    callbacks = callbacks
    )

train(callbacks=my_callbacks)
model.save_weights('model.h5')

model.predict(ap.chordBuilder(samples[440], notes))

np.reshape(ap.chordBuilder(samples[440], notes),(1,4410)).shape
