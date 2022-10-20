import tensorflow as tf
import random
import numpy as np
import model as mod
import notation

sampleRate = 44100
audioBufferLength = 100
Tuning = 440

'''
def generateSamples(duration, frequency, sampleRate):
    samples = int(sampleRate*duration)
    t = np.linspace(0, duration, samples)
    return np.sin(2*np.pi*frequency*t)


def generateTestingData(Tuning, sampleRate, audioBufferLength):
    normalizedTuning = Tuning/440 - 0.5
    data = []
    note, octave, frequency, interval = notation.generateNote(Tuning)
    data.append(np.insert(generateSamples(audioBufferLength/1000, frequency, sampleRate), 0, normalizedTuning))
    return np.array(data), frequency, note, interval, octave
'''
'''
def generateNote(ATuning):

    n = int(random.uniform(0,107)-57)
    frequency = ATuning * (2**(n/12))
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    startingInterval = 10
    startingOctave = 4
    octave = startingOctave + int(n/12)
    interval = startingInterval + (n - (int(n/12)*12))
    if interval > 12:
        interval -= 12
        octave += 1
    if interval <= 0:
        interval += 12
        octave -= 1
    return (notes[interval-1], octave, frequency, interval-1)
'''
TestingData, frequenciesGen, notesGen, intervalsGen, octavesGen = notation.generateTestingData(Tuning, sampleRate, audioBufferLength, 6)

'''
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2,activation='relu', input_shape=(None, int(sampleRate*(audioBufferLength/1000))+1)),
    tf.keras.layers.Dense(2,activation='relu'),
    tf.keras.layers.Dense(2,activation='relu'),
    tf.keras.layers.Dense(96, activation='sigmoid')
]
)
'''

model = mod.createModel(sampleRate, audioBufferLength)

model.load_weights('model.h5')
predictions = model.predict(TestingData)
predictionsAdjusted = [1 if i >= 0.5 else 0 for i in predictions[0]]
notesDetected = []
notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
for i in range(len(predictionsAdjusted)):
    if predictionsAdjusted[i] == 1:
        intervalInner = (i+1)%12
        octaveInner = int((i+1)/12)
        frequencyInner = 16.35 * (2**(i/12))
        notesDetected.append([notes[intervalInner-1],intervalInner,octaveInner,frequencyInner, i])
print(predictions[0])
maxes = np.argmax(predictions, axis=1)
maxDetected = []
if len(maxes) == 1:
    intervalInner = (maxes[0]+1)%12
    octaveInner = int((maxes[0]+1)/12)
    frequencyInner = 16.35 * (2**(maxes[0]/12))
    maxDetected.append([notes[intervalInner-1],intervalInner,octaveInner,frequencyInner, 0])
else:
    for i in range(len(maxes)):
        intervalInner = (maxes[i]+1)%12
        octaveInner = int((maxes[i]+1)/12)
        frequencyInner = 16.35 * (2**(maxes[i]/12))
        maxDetected.append([notes[intervalInner-1],intervalInner,octaveInner,frequencyInner, i])
accuracy = 0.8
accuracyWeightedIndexes = np.argwhere(predictions[0] > 0.5)
accDetected = []
if len(accuracyWeightedIndexes) == 1:
    intervalInner = (accuracyWeightedIndexes[0]+1)%12
    octaveInner = int((accuracyWeightedIndexes[0]+1)/12)
    frequencyInner = 16.35 * (2**(accuracyWeightedIndexes[0]/12))
    accDetected.append([notes[intervalInner-1],intervalInner,octaveInner,frequencyInner, 0])
else:
    for i in range(len(accuracyWeightedIndexes)):
        intervalInner = (accuracyWeightedIndexes[i]+1)%12
        octaveInner = int((accuracyWeightedIndexes[i]+1)/12)
        frequencyInner = 16.35 * (2**(accuracyWeightedIndexes[i]/12))
        accDetected.append([notes[intervalInner-1],intervalInner,octaveInner,frequencyInner, i])

print(predictionsAdjusted)
print(maxes)
print(f" Notes Detected were {notesDetected}: Format is Note, Interval, Octave, Frequency")
print(f"Max detected was {maxDetected}")
print(f"Detected above accuracy threshold was {accDetected}")
Notes = []
for i in range(len(notesGen)):
    Notes.append(f'Note: {notesGen[i]}{octavesGen[i]} Freq: {frequenciesGen[i]} Interval:{intervalsGen[i]}')

print(f"Actual Notes generated {Notes}")