import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
from zeta9.tools import coeffs_to_complex

def PlotV(ax, path, f=None, nmax=400, color="k", label=None, theta=1.576):
    d = np.load(path)

    v = d["dist"][:nmax]

    if(len(v) == 0):
        print(path, " "*(0 if len(path)>32 else 32-len(path)), "err=---------")
        return [], []

    print(path, " "*(0 if len(path)>32 else 32-len(path)), "err=%8.3e"%(v[0]))
    print(d["Y"][0])
    print(d["u_coeffs"][0])

    if(f):
        z9 = np.exp(2j*np.pi/9)
        a9 = (z9+1/z9).real
        u2 = np.array([1,1,0])
        for i in range(3):
            Y1 = d["Y"][0][i]
            Y = Y1[0] + Y1[1]*a9 + Y1[2]*a9**2
            print(Y1,np.abs(Y1/3**(2*f)-u2[i]))
        #
        u = np.array([np.exp(-0.5j * theta), -1, 0])
        for i in range(3):
            c1 = d["u_coeffs"][0][i]
            v1 = coeffs_to_complex(c1,f)
            print(v1,np.abs(u[i]-v1))

    #
    
    y = 1 + np.arange(len(v))

    ax.plot(v,y,ls="-",lw=2,ds='steps-pre',color=color,label=label)
#


fig, ax = plt.subplots(figsize=(8, 6))

ax.set_xscale("log")
ax.set_xlabel(r"$\epsilon$")
ax.set_ylabel(r"N(<$\epsilon)$")


PlotV(ax,"D/R2_f=2_eps=1e-2.npz",color="r",label=r"$|u|^2=2, f_V=2$")
PlotV(ax,"D/R2_f=3_eps=4e-4.npz",color="g",label=r"$|u|^2=2, f_V=3$")
PlotV(ax,"D/R2_f=4_eps=4e-5.npz",color="b",label=r"$|u|^2=2, f_V=4$")

PlotV(ax,"D/R1_f=3_eps=3e-3.npz",color="orange",label=r"$|u|^2=1, f_V=3$")

#ax.legend()
fig.tight_layout()
plt.show()


