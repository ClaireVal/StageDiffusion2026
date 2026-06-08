#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:18:36 2026

@author: valenchonc
"""

import numpy.random as rd
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import exp
from scipy import constants
import numpy as np

#mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#mpl.rc('text', usetex=True)
mpl.rcParams['ytick.labelsize']=30
mpl.rcParams['xtick.labelsize']=30
#mpl.rcParams['text.fontsize']=20

mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['figure.figsize'] = (12, 12)

# Taille de l'espace considéré
H = 100
L = 100
Natomes = H*L

#Choix des probas associées à chaque déplacement (gauche, droite, haut, bas)
g=0.25
d=0.25
h=0.25
b=0.25

#Constantes utiles
kB = constants.k

### Paramètres du système
# L'ordre de grandeur pour les énergies de formation et de migration d'un défaut sont de l'ordre de 1eV (cf rapport de stage de Manon Dewynter)
G_M = 1.5 #eV = barrière d'énergie pour l'échange lacune / atome du réseau
G_Mdef = 0.5 #eV = barrière d'énergie pour l'échange lacune / atome substitué
G_F = 1.0 #eV = énergie libre de Gibbs de formation du défaut substitutionnel
T = 300 #K
nu_0 = 10e13

# Positions aléatoires des défauts
# cDef = exp(-G_F/(kB*T))
# Ndef = floor(cDef*N)
Ndef = 10
cDef = Ndef/Natomes

listeDefauts = [(i,80) for i in range(5, min(H,L)-5,20)]        #exemple de positions de défauts ordonnées

# for i in range(Ndef):
#     u = rd.random()
#     v = rd.random()
#     xD = int(u*L)
#     yD = int(v*H)
#     listeDefauts.append((xD,yD))

print("Positions des défauts : ", listeDefauts)
print("")

defauts = set(listeDefauts)

### Initialisation des pas
gauche = 0
droite = 0
haut = 0
bas = 0

N=1000000
Densite = np.zeros((L, H))

positions = np.zeros((N+1,2))
x0 = [0,0]
x=x0.copy()


# Fonction permettant de connaître l'environnement, s'il y a des défauts dans cet environnement, et si oui où
def environnement(x):
    env = [(x[0], (x[1]-1)%H), (x[0], (x[1]+1)%H), ((x[0]-1)%L, x[1]), ((x[0]+1)%L, x[1])]          #on considère les 4 plus proches voisins
    testDef = False
    voisinsDef = []
    for a in env:
        if a in defauts:
            testDef = True
            voisinsDef.append(a)
    return (env, testDef, voisinsDef)
    

### Marche aléatoire

for i in range(1,N+1):
    u = rd.random()
    env = environnement(x)
    gamma = np.zeros(4)                     #Vecteur de dimension 4 donnant les fréquences de saut vers les directions [droite, haut, gauche, bas], chaque direction correspondant à un indice précis
    if env[1]==True:                        #On teste à chaque pas la présence de défauts, et retenons où ils sont
        envDef = env[2]
        for df in envDef:                    #Calcul des différentes fréquences de sauts des plus proches voisins
                if x[0]==df[0]:
                    if x[1]<df[1]:
                        gamma[1] = nu_0 * exp(-G_Mdef/(kB*T))
                    else:
                        gamma[3] = nu_0 * exp(-G_Mdef/(kB*T))
                else:
                    if x[0]<df[0]:
                        gamma[2] = nu_0 * exp(-G_Mdef/(kB*T))
                    else:
                        gamma[0] = nu_0 * exp(-G_Mdef/(kB*T))
                    
        for i in range(len(gamma)):
            if gamma[i]==0:
                gamma[i] = nu_0 * exp(-G_M/(kB*T))
                
        u *= np.sum(gamma)
        if u < gamma[0]:
            x[0] += 1
            droite += 1
            positions[i] = x.copy()
            x[0] = x[0]%L
            Densite[x[0], x[1]]+=1
                    
        elif u < np.sum(gamma[:2]):
            x[1] += 1
            haut += 1
            positions[i] = x.copy()
            x[1] = x[1]%H
            Densite[x[0], x[1]]+=1
            
        elif u < np.sum(gamma[:3]):
            x[0] -= 1
            gauche += 1
            positions[i] = x.copy()
            x[0] = x[0]%L
            Densite[x[0], x[1]]+=1
            
        else:
            x[1] -= 1
            bas += 1
            positions[i] = x.copy()
            x[1] = x[1]%H
            Densite[x[0], x[1]]+=1
        
        
        
    else:                                   #S'il n'y a pas de défaut dans l'environnement, c'est une marche aléatoire usuelle
        if u < g:
            x[0] -= 1
            gauche += 1
            positions[i] = x.copy()
            x[0] = x[0]%L
            Densite[x[0], x[1]]+=1
        
        elif u < g + d:
            x[0] += 1
            droite += 1
            positions[i] = x.copy()
            x[0] = x[0]%L
            Densite[x[0], x[1]]+=1
            
        elif u < g + d + h:
            x[1] += 1
            haut += 1
            positions[i] = x.copy()
            x[1] = x[1]%H
            Densite[x[0], x[1]]+=1
            
        else:
            x[1] -= 1
            bas += 1
            positions[i] = x.copy()
            x[1] = x[1]%H
            Densite[x[0], x[1]]+=1
        
print("(g,d,h,b)=", g,d,h,b, "   proportion de gauche/droite/haut/bas expérimentales=", gauche/(gauche+droite+haut+bas), droite/(gauche+droite+haut+bas), haut/(gauche+droite+haut+bas), bas/(gauche+droite+haut+bas))
fig, ax = plt.subplots()
# abscisses = [elem[0] for elem in positions]
# ordonnees = [elem[1] for elem in positions]

# ax[0].scatter(abscisses, ordonnees, c=range(len(abscisses)), cmap='viridis')
# ax[0].set_title(f"Marche aléatoire 2D (g={g}, d={d}, h={h}, b={b})", fontsize=16)
# ax[0].set_xlabel("Position horizontale", fontsize=14)
# ax[0].set_ylabel("Position verticale", fontsize=14)
# ax[0].tick_params(axis='both', labelsize=14)


im = ax.imshow(Densite.T, cmap='plasma', origin='lower')
fig.colorbar(im, ax=ax)
ax.set_xlabel('X', fontsize=14)
ax.set_ylabel('Y', fontsize=14)
ax.set_title(f"Mapping de la densité de passage\nposition initiale {x0}", fontsize=16)
ax.tick_params(axis='both', labelsize=14)

XDisplay = []
YDisplay = []
for (a,b) in listeDefauts:
    XDisplay.append(a)
    YDisplay.append(b)

plt.plot(XDisplay, YDisplay, "^r")
plt.show()                         