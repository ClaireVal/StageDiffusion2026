#Ce code permet de calculer le MSD d'une marche aléatoire, avec une pondération selon les STD.
#Deux cas sont traités et calculés: Une longue trajectoire vs plusieurs trajectoires plus courtes.
#On se place en échelle log-log pour plus de précision, mais des échelles linéaires sont également possibles. Ce code peut facilement être adapté à une distribution linéaire en remplaçant les logspace par des linspace.

import numpy.random as rd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from math import exp

#mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#mpl.rc('text', usetex=True)
#mpl.rcParams['ytick.labelsize']=30
#mpl.rcParams['xtick.labelsize']=30
#mpl.rcParams['text.fontsize']=20
mpl.rcParams['axes.labelsize']=35
mpl.rcParams['legend.fontsize']=30
mpl.rcParams['axes.titlesize']=35
mpl.rcParams['figure.figsize']=(12,12)

# Choix d'un p fixe
p = 0.5
# Initialisation des pas
N= 1000
M = 100



# Marche 1D d'une trajectoire de M*N pas réguliers

positionsLongue = np.zeros(M*N)
positionsLongue[0]=0
x = 0

for i in range(1, M*N):
    u = rd.random()
    if u<p:
        positionsLongue[i]=x-1
        x-=1
    else:
        positionsLongue[i]=x+1
        x+=1


# Marche de M trajectoires de N pas réguliers

Mpositions = []

for k in range(M):
    positions = np.zeros(N)
    positions[0]=0
    x = 0
    for i in range(1,N):
        u = rd.random()
        if u<p:
            positions[i]=x-1
            x-=1
        else:
            positions[i]=x-1
            x+=1
    Mpositions.append(np.copy(positions))




###  Calcul du MSD  ###

# Premier cas : Moyenne sur une grande trajectoire de M*N points

def MSDLongue(x,dt):
    n = M*N - dt                         #nombre de (t1,t2) tels que t2-t1=dt
    sum1 = 0
    sum2 = 0
    for i in range(n):
        sum1 += (x[i+dt]-x[i])**2
        sum2 += (x[i+dt]-x[i])**4       #on considère les x de la k-ième trajectoire, et au i-ième pas de temps
    return sum1/n, np.sqrt(((sum2/n) - (sum1/n)**2)/n)

# Second cas : Moyenne sur M trajectoires (à parallèliser)

def MSDCourtes(x, dt, M):
    #x a la dimension M*N
    n = N-dt                          #nb de pas de temps à considérer
    sum1 = 0
    sum2 = 0
    for k in range(n):                          #on parcourt les M trajectoires à t fixé
        for i in range(M):                      #on parcourt les M trajectoire
            sum1 += (x[i][k+dt]-x[i][k])**2       #on considère les x de la k-ième trajectoire, et au i-ième pas de temps
            sum2 += (x[i][k+dt]-x[i][k])**4       #on considère les x de la k-ième trajectoire, et au i-ième pas de temps
    return sum1/(M*n), np.sqrt(((sum2/(M*n)) - (sum1/(M*n))**2)/(M*n))


# Définition des différents pas de temps
dtLogListe_L = np.unique(np.round(np.logspace(0, np.log10((M*N)//10), 200)).astype(int))
dtLogListe_C = np.unique(np.round(np.logspace(0, np.log10((N)//10), 200)).astype(int))        #np.int_(np.logspace(0, np.log10(N//10), 100))
    
MSDLog_Long = np.zeros(len(dtLogListe_L))
MSDLog_Cour = np.zeros(len(dtLogListe_C))
stdLogLong = np.zeros(len(dtLogListe_L))
stdLogCour = np.zeros(len(dtLogListe_C))

# Calcul des MSD correspondants
for ind, l in enumerate(dtLogListe_L):
    MSDLog_Long[ind] = MSDLongue(positionsLongue, l)[0]
    stdLogLong[ind] = MSDLongue(positionsLongue, l)[1]
for ind, c in enumerate(dtLogListe_C):
    MSDLog_Cour[ind] = MSDCourtes(Mpositions, c, M)[0]
    stdLogCour[ind] = MSDCourtes(Mpositions, c, M)[1]
 

#Il est possible, à ce niveau, de sauvegarder les MSD afin de les analyser dans un autre fichier. Cela permet de faire différentes analyses en gagnant du temps de calcul.
#np.savez("MSD.npz", dtLogListe_L = dtLogListe_L, dtLogListe_C=dtLogListe_C, MSDLog_Long=MSDLog_Long, MSDLog_Cour=MSDLog_Cour, stdLogLong=stdLogLong, stdLogCour=stdLogCour)
# #Les données sont alors récupérées ainsi :
# data = np.load("MSD.npz")
# dtLogListe_L = data["dtLogListe_L"]
# dtLogListe_C=data["dtLogListe_C"]
# MSDLog_Long=data["MSDLog_Long"]
# MSDLog_Cour=data["MSDLog_Cour"]
# stdLogLong=data["stdLogLong"]
# stdLogCour=data["stdLogCour"]


###  Fitting du MSD par une fonction linéaire  ###

# Definition de la fonction linéaire pour le fit
def lineaire(t, x, y):
    return x + t*y


#Fitting en échelle log/log avec une pondération selon les STD
print("MIN/MAX des STD (trajectoire longue) : ", np.min(stdLogLong), np.max(stdLogLong))         #Il y a des 0 dans la liste qui faussent la pondération en donnant des poids énormes à certains points
stdLogLong[stdLogLong == 0] = 1e-8                                                               #==> On les remplace par une petite valeur non nulle
stdLogCour[stdLogCour == 0] = 1e-8

poptL4, _ = curve_fit(lineaire, np.log(dtLogListe_L), np.log(MSDLog_Long), sigma = stdLogLong)
poptC4, _ = curve_fit(lineaire, np.log(dtLogListe_C), np.log(MSDLog_Cour), sigma = stdLogCour)

D_L4 = exp(poptL4[0])/2
D_C4 = exp(poptC4[0])/2

print("Coefficient de diffusion cas trajectoire longue avec pondération std : D_L = ", D_L4)
print("Coefficient de diffusion cas M trajectoires courtes avec pondération std : D_C = ", D_C4)


#Tracé pour la pondération STD
fig, ax = plt.subplots(2) 
ax[0].plot(dtLogListe_L, MSDLog_Long, 'o',color = "red")
ax[0].set_xlabel("tau", fontsize=14)
ax[0].set_ylabel("MSD", fontsize=14)
ax[0].set_title(f"MSD de la longue trajectoire en fonction de tau\n D_L = {D_L4}\n avec pondération selon STD", fontsize=14)
ax[0].set_xscale("log")
ax[0].set_yscale("log")
yFitL = poptL4[1]*dtLogListe_L+poptL4[0]
ax[0].plot(dtLogListe_L, yFitL, label="Courbe fittée trajectoire longue", color="orange",ls='--',lw=2)
ax[1].plot(dtLogListe_C, MSDLog_Cour, 'o',color = "red")
ax[1].set_xlabel("tau", fontsize=14)
ax[1].set_ylabel("MSD", fontsize=14)
ax[1].set_title(f"MSD des courtes trajectoires en fonction de tau\n D_C = {D_C4}\n avec pondération selon STD", fontsize=14)
ax[1].set_xscale("log")
ax[1].set_yscale("log")
yFitC = poptC4[1]*dtLogListe_C+poptC4[0]
ax[1].plot(dtLogListe_C, yFitC, label="Courbe fittée trajectoires courtes", color="orange",ls='--',lw=2)
ax[0].tick_params(axis='both', labelsize=12)
ax[1].tick_params(axis='both', labelsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.subplots_adjust(hspace = 0.5)     
