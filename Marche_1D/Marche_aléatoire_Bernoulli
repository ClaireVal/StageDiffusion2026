#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:22:43 2026

@author: valenchonc
"""

import numpy.random as rd
import matplotlib.pyplot as plt
import matplotlib as mpl

#mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#mpl.rc('text', usetex=True)
mpl.rcParams['ytick.labelsize']=30
mpl.rcParams['xtick.labelsize']=30
#mpl.rcParams['text.fontsize']=20
mpl.rcParams['axes.labelsize']=35
mpl.rcParams['legend.fontsize']=30
mpl.rcParams['axes.titlesize']=35
mpl.rcParams['figure.figsize']=(12,12)

#Choix d'un p fixe
p = 0.5
#Initialisation des pas
left = 0
right = 0
positions = [0]
x = 0
N=10000

#Loi de bernoulli
for i in range(N):
    u = rd.random()
    if u<p:
        left+=1
        positions+=[x-1]
        x-=1
    else:
        right+=1
        positions+=[x+1]
        x+=1

print("p=", p, "  (left,right)=",(left, right), "  proportion de gauche=", left/(left+right))

fig, ax = plt.subplots()             # Create a figure containing a single Axes.
ax.plot(positions, [i for i in range(N+1)])  # Plot some data on the Axes.
ax.set_title(f"Marche aléatoire selon une loi de Bernoulli de paramètre p={p}", fontsize=16)
ax.set_xlabel("Position", fontsize=14)
ax.set_ylabel("Temps", fontsize=14)
mpl.rcParams['axes.labelsize']=14
mpl.rcParams['legend.fontsize']=14
mpl.rcParams['axes.titlesize']=14
mpl.rcParams['figure.figsize']=(12,12)
plt.tick_params(axis='both', labelsize=14)
plt.show()                           # Show the figure.
