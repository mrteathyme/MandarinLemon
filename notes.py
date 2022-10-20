import random
def generateNote():

    n = int(random.uniform(0,117)-57)
    frequency = 440 * (2**(n/12))
    intervalsPerOctave = 12
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    startingInterval = 10
    startingOctave = 4
    octave = startingOctave + int(n/12)
    interval = startingInterval + (n - (int(n/12)*12))
    if interval > 12:
        interval -= 12
        octave += 1
    if interval == 0 or -11:
        octave -= 1
    return (notes[interval-1], octave, frequency)

print(generateNote())