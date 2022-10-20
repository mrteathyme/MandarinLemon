from scipy.io.wavfile import read, write
import numpy as np
import os



def loadData():
    pitchSamples = {}
    chunkSize = 4410
    for filename in os.listdir('./TestData'):
        filenameStripped = os.path.splitext(filename)[0]
        data = read(f"./TestData/{filenameStripped}.wav")

        Tuning, Octave, Pitch = filenameStripped.split('_')
        if int(Tuning) not in pitchSamples.keys():
            pitchSamples[int(Tuning)] = {}
        if int(Octave) not in pitchSamples[int(Tuning)].keys():
            pitchSamples[int(Tuning)][int(Octave)] = {}
        ar = np.array(data[1],dtype=float)
        pitchSamples[int(Tuning)][int(Octave)][Pitch] = []
        dataTrimmed = np.trim_zeros(np.trim_zeros(ar / np.iinfo(np.int16).max, 'f'),'b')
        for i in range(int(len(dataTrimmed)/chunkSize)-1):
            pitchSamples[int(Tuning)][int(Octave)][Pitch].append(dataTrimmed[int(i*chunkSize):int((i+1)*chunkSize)])
    return pitchSamples
#print(pitchSamples)



def chordBuilder(pitchSamples, notes):
    chord = []
    for i in range(len(notes)):
        chord.append(pitchSamples[notes[i][0]][notes[i][1]][notes[i][2]])
    return np.array(sum(chord))



'''
outfile = "sounds.wav"
notes = []
notes.append([2,'A',2])
notes.append([2,'B',2])
notes.append([3,'C',2])
chordBuilder(pitchSamples[440],notes)
write(outfile,44100,(chordBuilder(pitchSamples[440],notes) * np.iinfo(np.int16).max).astype(np.int16))
'''