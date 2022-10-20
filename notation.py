import random
import numpy as np

def generateSamples(duration, frequency, sampleRate):
    samples = int(sampleRate*duration)
    t = np.linspace(0, duration, samples)
    return np.sin(2*np.pi*frequency*t)

def generateTestingData(Tuning, sampleRate, audioBufferLength, maxChordNotes):
    normalizedTuning = Tuning/440 - 0.5
    data = []
    notes = []
    octaves = []
    intervals = []
    frequencies = []
    if random.choice([True,False]):
        chordNotes = random.randrange(3,maxChordNotes+1)-1
        for i in range(chordNotes):
            note, octave, frequency, interval = generateNote(Tuning)
            notes.append(note)
            octaves.append(octave)
            intervals.append(interval)
            frequencies.append(frequency)
        frequency = sum(frequencies)/chordNotes
    else:
        note, octave, frequency, interval = generateNote(Tuning)
        notes = [note]
        octaves = [octave]
        intervals = [interval]
        frequencies = [frequency]
    data.append(np.insert(generateSamples(audioBufferLength/1000, frequency, sampleRate), 0, normalizedTuning))
    return np.array(data), frequencies, notes, intervals, octaves


def generateTrainingData(num, sampleRate, audioBufferLength, maxChordNotes):
    Tuning = random.uniform(432,450)
    normalizedTuning = Tuning#Tuning/440 - 0.5
    data = []
    expectedOutput = []
    for i in range(num):
        #note, octave, frequency, interval = generateNote(Tuning)
        #data.append(np.insert(generateSamples(audioBufferLength/1000, frequency, sampleRate), 0, normalizedTuning))
        notes = []
        octaves = []
        intervals = []
        frequencies = []
        if random.choice([True,False]):
            chordNotes = random.randrange(3,maxChordNotes+1)-1
            for i in range(chordNotes):
                note, octave, frequency, interval = generateNote(Tuning)
                notes.append(note)
                octaves.append(octave)
                intervals.append(interval)
                frequencies.append(frequency)
            frequency = sum(frequencies)/chordNotes
        else:
            note, octave, frequency, interval = generateNote(Tuning)
            notes = [note]
            octaves = [octave]
            intervals = [interval]
            frequencies = [frequency]
        data.append(np.insert(generateSamples(audioBufferLength/1000, frequency, sampleRate), 0, normalizedTuning))
        expectedOutputInner = []
        for y in range(108):
            if y in intervals:
                expectedOutputInner.append(1)
            else:
                expectedOutputInner.append(0)
        expectedOutput.append(np.array(expectedOutputInner))
    return (np.array(data), np.array(expectedOutput))

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