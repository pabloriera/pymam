from pymam import *                  # importa todas las funciones de pymam
 
fs = 44100                           # frecuencia de sampleo, samples x segundo
dur = 0.5                            # duracion en segundos
 
t = time_array(0,dur,fs)
 
x = 0.8 * sin( 440 * 2*pi * t)       # sinusoide de 440 hz en el array x
 
sound(x, fs)                         # reproduce el array x
 
# para la reproduccion stereo se concatenan dos arrays
 
xl = 0.5*sin(440*2*pi*t)             # sinusoide de 440 hz en el array xl
xr = 0.5*sin(660*2*pi*t)             # sinusoide de 660 hz en el array xr
x = vstack((xl,xr))                  # combinacion de los dos arrays xl y xr en uno solo stereo
 
sound(x,fs)                          # reproduce el array x