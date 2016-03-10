# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 18:14:51 2014

@author: miles
"""

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm,hsv_to_rgb

# Data manipulation:

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


# Interface to LineCollection:

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc

def imagesc(*args,**kwargs):
    if len(args)==1:
        I = args[0]
        handle = plt.imshow(I,aspect='auto',interpolation='none',**kwargs)
        return handle
    if len(args)==3:
        I = args[2]
        x_start = args[0][0]
        x_end = args[0][-1]
        y_start = args[1][0]
        y_end = args[1][-1]
        extent=[x_start, x_end , y_start , y_end ]
        handle = plt.imshow(I,aspect='auto',interpolation='none', extent = extent,**kwargs)
        return handle
    if len(args)==2:
        print "Input should be imagesc(x,y,I) or imagesc(I)"

def axes3d():
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.gcf()
    if plt.gca().__class__.__name__=='AxesSubplot':
	ax = fig.add_subplot(*plt.gca().properties()['geometry'], projection='3d')
    elif plt.gca().__class__.__name__=='Axes3DSubplot':
	ax = plt.gca()
    else:
    	ax = fig.add_subplot(111, projection='3d')

    return ax

def plot3(x,y,z,*args,**kwargs):

    ax = axes3d()
    
    ax.plot3D(x,y,z,*args,**kwargs)
    plt.draw()
    
    return ax
    
def surf(X,Y,Z,*args,**kwargs):
    
    ax = axes3d()
    ax.plot_surface(X,Y,Z,*args,**kwargs)
    
    return ax

def scatter3(x,y,z,*args,**kwargs):
    ax = axes3d()
    
    ax.scatter3D(x,y,z,*args,**kwargs)
    plt.draw()

    return ax

def print_list(lst):
    print('\n'.join('{}: {}'.format(*k) for k in enumerate(lst))) 


def scattergroup(x,y,groups,colors = None,label=None, kwargs_group = None,**kwargs):
    
    group_ix = list(np.unique(groups))
          
    if type(colors)==type(None):
        colors = plt.cm.jet(np.linspace(0, 1, len(group_ix)))
      
    for ix,c in zip(group_ix,colors):

        if type(kwargs_group)!=type(None):
            args = dict(kwargs_group[ix].items()+kwargs.items())
        else:
            args = kwargs

        if type(label)==type(None): 
            la = ix
        else:
            la = label[ix]

        fi = pl.find(np.array(groups)==ix)
        x_ = x[fi]
        y_ = y[fi]
        pl.scatter(x_,y_,c=c,label=la,**args)

def scattergroup3(x,y,z,groups,colors = None,label=None, kwargs_group = None,**kwargs):
    
    group_ix = list(np.unique(groups))
          
    if type(colors)==type(None):
        colors = plt.cm.jet(np.linspace(0, 1, len(group_ix)))
    axes = []
    
    for ix,c in zip(group_ix,colors):

        if type(kwargs_group)!=type(None):
            args = dict(kwargs_group[ix].items()+kwargs.items())
        else:
            args = kwargs

        if type(label)==type(None): 
            la = ix
        else:
            la = label[ix]

        fi = pl.find(np.array(groups)==ix)
        x_ = x[fi]
        y_ = y[fi]
        z_ = z[fi]
        axes.append( scatter3(x_,y_,z_,c=c,label=la,**args) )

    return axes
    
def complexfunction(func,xl,yl):
    
    x,y = np.meshgrid(xl,yl)
    z = x + 1j*y
    
    H = func(z)
    
    C = np.ones((len(xl),len(yl),3))
    hue = (np.angle(H)+np.pi)/(2*np.pi)
    sat = np.exp(-abs(H)/abs(H).max()*4)
    val = np.exp(-sat/2)
    C[:,:,0] = hue
    C[:,:,1] = sat
    C[:,:,2] = val
    
    pl.imshow(hsv_to_rgb(C),extent= [min(xl),max(xl),min(yl),max(yl)]);
        
def plot_cov_ellipse(cov, pos, volume=.5, ax=None, fc='none', ec=[0,0,0], a=1, lw=2):
    """
    Plots an ellipse enclosing *volume* based on the specified covariance
    matrix (*cov*) and location (*pos*). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        volume : The volume inside the ellipse; defaults to 0.5
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
    """

    import numpy as np
    from scipy.stats import chi2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    kwrg = {'facecolor':fc, 'edgecolor':ec, 'alpha':a, 'linewidth':lw}

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(chi2.ppf(volume,2)) * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)

    ax.add_artist(ellip)

def labelsubplots(loc=0,st=''):
    from string import ascii_lowercase
    fig = pl.gcf() 
    fs = pl.rcParams['font.size']
    if loc==0:
        locc = [0.985,1-fs/120.0]
        align = 'right'
    else:
        locc = [0.015,1-fs/120.0]
        align = 'left'

    for i,ax in enumerate(fig.axes):
        if st=='':
            s = '('+ascii_lowercase[i]+')'
        else:
            s = st[i]
        ax.text(locc[0],locc[1],s,horizontalalignment=align,transform=ax.transAxes)


def pairwise_significance(coords,bools,height=1,wide=0.5,type=1):
    
    N = len(coords)
    pares = [[i,j] for i in range(N) for j in range(i+1,N)]
    
    marker_list = ['*','v','d','s','<','p','^']
#    (u'o', u'v', u'^', u'<', u'>', u'8', u's', u'p', u'*', u'h', u'H', u'D', u'd')
    k = 0
    for (b,(i,j)) in zip(bools,pares):       
        if b :
            h = height+wide*k/float(N)
            if type:
                pl.plot([coords[i],coords[j]],[h,h],'|-k',markersize=10,lw=2)
            else:
                pl.plot([coords[i],coords[j]],[h,h],marker_list[k],markersize=10 )
            k+=1
        
    
def xylabel(xlabel,ylabel,bottom=0.02,left = 0.05):
    pl.text(left,0.5,ylabel,verticalalignment='center',rotation='vertical',horizontalalignment='center',transform=pl.gcf().transFigure)
    pl.text(0.5,bottom,xlabel,horizontalalignment='center',transform=pl.gcf().transFigure)


def multiplot(x,y,row=True):
    
    Nsp = len(y.keys())
    
    if len(x.keys())==1:
        xk = x.keys()[0]
        
        for i,yk in enumerate(y.iterkeys()):
            if row:
                pl.subplot(Nsp,1,i+1)
            else:
                pl.subplot(1,Nsp,i+1)
    
            pl.plot(x[xk],y[yk],label=yk)
            pl.xlabel(xk)
            pl.ylabel(yk)
        

    else:
    
        for i,(xk,yk) in enumerate(zip(x.iterkeys(),y.iterkeys())):
            if row:
                pl.subplot(Nsp,1,i+1)
            else:
                pl.subplot(1,Nsp,i+1)
    
            pl.plot(x[xk],y[yk],label=yk)
            pl.xlabel(xk)
            pl.ylabel(yk)           

    pl.tight_layout()  

