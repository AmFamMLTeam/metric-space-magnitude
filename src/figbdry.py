#!/usr/bin/env python3.7
import matplotlib 
matplotlib.use('Agg')
import pylab as pl
from matplotlib import rc
rc('text', usetex=True)


import numpy as np
import numpy.linalg as nl
import numpy.random as nr
import os.path
from scipy.optimize import minimize
import sys

basename = os.path.splitext(sys.argv[0])[0]
pdfname = basename + '.pdf'
pngname = basename + '.png'
epsname = basename + '.eps'
n = 15000
r=50.000
# r=0.10

def mkmat(x):
    ret = np.zeros( (len(x), len(x)))
    
    for i,xi in enumerate(x):
        for j,xj in enumerate(x):
            ret[i,j]= np.exp(-r*nl.norm(xi-xj))
    return ret

X0 = nr.random((n,2))
X = [_ for _ in X0 if 
       (nl.norm(_-(0.25))<0.15 and nl.norm(_-0.28)>0.07) or 
       (nl.norm(_-0.6)<0.25 and nl.norm(_-0.7)>0.25 ) or 
     (_[0]>=0.55 and (1-0.05*np.sin(0.123+_[1]*2*np.pi*5))*_[0]<=0.8 and _[1]<=0.32 and _[1]>=0.05) or
     (0.0< (_[0]-_[1])+0.6< 0.02)
      ]

M = mkmat(X)
IM = nl.inv(M)
print(sum(sum(IM)))

Xx = [_[0] for _ in X]
Xy = [_[1] for _ in X]

fig = pl.figure(figsize=(5,5))
ax = pl.subplot(111)
C = IM.dot(np.ones(len(X)))
sc = ax.scatter(Xx, Xy, c=C, s = (20*abs(C))**2.0, lw=0)
ax.set_aspect('equal')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])


# ax.set_title(r'$\#X=%d,\, e^{-%0.3f \|x\|}$' % (len(X), r))
pl.colorbar(sc, shrink=0.8)
pl.tight_layout()
pl.savefig(pngname)
