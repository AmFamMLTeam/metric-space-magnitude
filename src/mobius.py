#!/usr/bin/env python3.7
import matplotlib 
matplotlib.use('Agg')
import pylab as pl
from matplotlib import rc
import matplotlib.cm as cm
from matplotlib.colors import Normalize

rc('text', usetex=True)

import numpy as np
import numpy.linalg as nl
import numpy.random as nr
import os.path
from numpy import cos, sin
import sys

basename = os.path.splitext(sys.argv[0])[0]
pdfname = basename + '.pdf'
pngname = basename + '.png'
n = 4000
r=25.000
tx=np.pi*0
ty=np.pi/3
tz=np.pi
Rx = np.array([[1, 0, 0], [0, cos(tx), -sin(tx)], [0, sin(tx), cos(tx)]])
Ry = np.array([[cos(ty), 0, sin(ty)], [0, 1, 0], [-sin(ty), 0, cos(ty)]])
Rz = np.array([[cos(tz), -sin(tz), 0],[sin(tz), cos(tz), 0], [0,0,1]])

def mkmat(x):
    ret = np.zeros( (len(x), len(x)))
    
    for i,xi in enumerate(x):
        for j,xj in enumerate(x):
            ret[i,j]= np.exp(-r*nl.norm(xi-xj))
    return ret

nr.seed(3)
X0 = nr.random((n,2))
X0[:,0]*=2*np.pi
X0[:,1]-=0.5
X0[:,1]*=2
X = np.array([((1+(t/2)*cos(s/2))*cos(s), (1+(t/2)*cos(s/2))*sin(s), t/2*sin(s/2)) for s,t in X0])
X = X@Rx@Ry@Rz

M = mkmat(X)
IM = nl.inv(M)
print(sum(sum(IM)))

Xx = [_[0] for _ in X]
Xy = [_[1] for _ in X]
Xz = [_[2] for _ in X]


fig = pl.figure(figsize=(6,3))
ax = fig.add_subplot(121, projection='3d')
# ax = fig.add_subplot(121)
bx = fig.add_subplot(122)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.axes.zaxis.set_ticklabels([])
ax.xaxis._axinfo["grid"]['linewidth'] = 0.1
ax.yaxis._axinfo["grid"]['linewidth'] = 0.1
ax.zaxis._axinfo["grid"]['linewidth'] = 0.1

C = IM.dot(np.ones(len(X)))
sa = 50*abs(C)**3
sb = sa
cmap = pl.get_cmap()
norm = Normalize(vmin=min(sa), vmax=max(sa))
for idx, row in enumerate(X):
    ax.plot(row[0], row[1], '.', color = cmap(norm(sa[idx])), markersize=sa[idx], markeredgecolor='none')

# ax.plot(Xx, Xy, Xz, c=C, markersize=sa/10, lw=0)
sc = bx.scatter(X0[:,0], X0[:,1], c=C, s=sb, lw=0)
xoffset=0.08
bx.arrow(0-xoffset, -1, 0, 2, length_includes_head=True, head_width=0.2, head_length=0.1, fc='0')
bx.arrow(2*np.pi+xoffset, 1, 0, -2, length_includes_head=True, head_width=0.2, head_length=0.1, fc='0')
bx.axis('off')
pl.colorbar(sc, shrink=0.9)
pl.savefig(pdfname)
pl.savefig(pngname, dpi=300)
