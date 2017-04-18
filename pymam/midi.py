# -*- coding: utf-8 -*-

"""
Midi
"""

import music21

__all__ = ["music21","mtof", "playscore", "playnote", "scoresequence", "playsequence", "midiwrite"]

def mtof(m): 
    """
    Midi to Frequency (Hz)
    """

    f = 440*(2**((m-69)/12.0))

    return f

try:
    from music21 import stream, instrument, note, duration, midi, tempo


    def playscore(s):
        midi.realtime.StreamPlayer(s).play()
        
    def playnote(notas,dur,instrumento = instrument.Piano()):

        from numpy import float64
        
        if type(notas)==str:
            s = stream.Score()
            p1 = stream.Part()
            p1.insert(instrumento)
            m1p1 = stream.Measure()
            nota = note.Note(notas,duration=duration.Duration(dur))
            m1p1.append(nota)
            p1.append(m1p1)
            s.insert(0,p1)
            playscore(s)
            
        elif isinstance(notas, (int, long, float, float64)):
            s = stream.Score()
            p1 = stream.Part()
            p1.insert(instrumento)
            m1p1 = stream.Measure()
            nota = note.Note(duration=duration.Duration(dur))
            nota.frequency = notas
            m1p1.append(nota)
            p1.append(m1p1)
            s.insert(0,p1)
            playscore(s)

        return nota

    def scoresequence(notes,durations,velocities=None,instrument = None,bpm=60):

        if instrument==None:
            instrument = music21.instrument.Piano()

        s = stream.Stream()
        s.append(tempo.MetronomeMark(number=bpm))
        s.append(instrument)

        for n,d in zip(notes,durations):
            if n==0:
                s.append(note.Rest(n,duration=duration.Duration(d)))
            else:
                s.append(note.Note(n,duration=duration.Duration(d)))
        
        if velocities==None:
            velocities=len(notes)*[127]
        
        for n,v in zip(s.flat.notes,velocities):
            n.volume = v
     
        sc = stream.Score()
        sc.insert(s)        
        
        return sc
        
    def playsequence(notas,durs,velocities=None,instrumento = instrument.Piano(),bpm=60):
        score = scoresequence(notas,durs,velocities=velocities,instrumento=instrumento,bpm=bpm)
        playscore(score)
        return score

    def midiwrite(score,name):
        midi_file = open(name,'wb')
        midi_file.write(midi.translate.streamToMidiFile(score).writestr())
        midi_file.close()

except:
    "No music21"
    pass
