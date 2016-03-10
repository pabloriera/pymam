# -*- coding: utf-8 -*-

"""
Midi
"""

__all__ = ["mtof"]

def mtof(m): 
    """
    Midi to Frequency (Hz)
    """

    f = 440*(2**((m-69)/12.0))

    return f


try:
    from music21 import stream, instrument, note, duration, midi,tempo
 
    from numpy import float64

    def playscore(s):
        midi.realtime.StreamPlayer(s).play()
        
    def playnote(notas,dur,instrumento = instrument.Piano()):
        
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

    def scoresequence(notas,durs,vols=None,instrumento = instrument.Piano(),bpm=60):
        
        s = stream.Stream() 
        s.append(tempo.MetronomeMark(number=bpm))
        for n,d in zip(notas,durs):
            if n==0:
                s.append(note.Rest(n,duration=duration.Duration(d)))
            else:
                s.append(note.Note(n,duration=duration.Duration(d)))
        
        if vols==None:
            vols=len(notas)*[127]        
        
        p = stream.Part()
        p.insert(instrumento)    
        p.append(s)
        sc = stream.Score()
        sc.insert(p)
        
        for n,v in zip(sc.flat.notes,vols):
            n.volume = v
     
        return sc
        
    def playsequence(notas,durs,vols=None,instrumento = instrument.Piano(),bpm=60):
        score = scoresequence(notas,durs,vols=vols,instrumento=instrumento,bpm=bpm)
        playscore(score)
        return score

    def midiwrite(score,name):
        midi_file = open(name,'wb')
        midi_file.write(midi.translate.streamToMidiFile(score).writestr())
        midi_file.close()
except:
    "No music21"
    pass