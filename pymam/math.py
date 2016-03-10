# -*- coding: utf-8 -*-

__all__ = ["fraction","mirror"]

def fraction(x,limit=0):

    """
    Devuelve el numerador y el denominador de una fraccion que aproxima al numero real de entrada.
    El segundo argumento permite limitar el valor del denominador    
    """
    from numpy import ndarray
    import fractions,decimal
    
    if type(x)== ndarray:

        y = x.flatten()        
        num = []; den = []
        for i in range(len(y)):

            if limit <1:    
                fra = fractions.Fraction(decimal.Decimal(str(y[i]))) 
                num.append(fra.numerator)
                den.append(fra.denominator)
            else:
                fra = fractions.Fraction(y[i]).limit_denominator(limit)
                num.append(fra.numerator)
                den.append(fra.denominator)
        
        return num,den
    
    else:
        if limit <1:    
            fra = fractions.Fraction(decimal.Decimal(str(x)))    
        else:
            fra =  fractions.Fraction(x).limit_denominator(limit)
        
        return fra.numerator, fra.denominator
    
def mirror(x):
    """
    Returns the mirror a list
    """
    
    aux = x[:]
    for i in range(len(x)):
        aux[i]=x[-i-1]
        
    return aux

