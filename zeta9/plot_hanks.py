import numpy as np
import matplotlib.pyplot as plt
import sys, os, glob


fig = plt.figure(figsize=(8,6))
fig.subplots_adjust(left=0.10,right=0.98,bottom=0.15,top=0.98,hspace=0,wspace=0.1)

ax = fig.add_subplot(1,1,1)


ax.set_xlim(0,10)
ax.set_xlabel(r"${\rm log}_10(1/\epsilon)$")
ax.set_ylim(0,10)
ax.set_ylabel(r"$2f_V$")


img = plt.imread("slope.png")
ax.imshow(img,extent=[0,10,0,10],aspect="auto",alpha=0.5)


def Plot(ax,color="orange"):
    f = np.array([2,3,4])
    eps = np.array([4.842e-03,2.632e-04,2.902e-05])

    x = np.log10(1/eps)
    y = 2*f

    ax.plot(x,y,"s",color=color,ms=16)
#    


Plot(ax)


plt.show()

