import numpy.random as rd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

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
N= 100000
M = 1



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
dtLinListe_L = np.int_(np.linspace(0, (M*N)//10, 1000))
dtLinListe_C = np.int_(np.linspace(0, N//10, 100))
dtLogListe_L = np.unique(np.round(np.logspace(0, np.log10((M*N)//10), 200)).astype(int))
dtLogListe_C = np.unique(np.round(np.logspace(0, np.log10((N)//10), 200)).astype(int))        #np.int_(np.logspace(0, np.log10(N//10), 100))
    
MSDLin_Long = np.zeros(len(dtLinListe_L))
MSDLin_Cour = np.zeros(len(dtLinListe_C))
MSDLog_Long = np.zeros(len(dtLogListe_L))
MSDLog_Cour = np.zeros(len(dtLogListe_C))
stdLinLong = np.zeros(len(dtLinListe_L))
stdLinCour = np.zeros(len(dtLinListe_C))
stdLogLong = np.zeros(len(dtLogListe_L))
stdLogCour = np.zeros(len(dtLogListe_C))

# Calcul des MSD correspondants
for ind, l in enumerate(dtLinListe_L):
    MSDLin_Long[ind] = MSDLongue(positionsLongue, l)[0]
    stdLinLong[ind] = MSDLongue(positionsLongue, l)[1]
for ind, c in enumerate(dtLinListe_C):
    MSDLin_Cour[ind] = MSDCourtes(Mpositions, c, M)[0]
    stdLinCour[ind] = MSDCourtes(Mpositions, c, M)[1]
for ind, l in enumerate(dtLogListe_L):
    MSDLog_Long[ind] = MSDLongue(positionsLongue, l)[0]
    stdLogLong[ind] = MSDLongue(positionsLongue, l)[1]
for ind, c in enumerate(dtLogListe_C):
    MSDLog_Cour[ind] = MSDCourtes(Mpositions, c, M)[0]
    stdLogCour[ind] = MSDCourtes(Mpositions, c, M)[1]
    
np.savez("MSD2.npz", dtLinListe_L = dtLinListe_L, dtLinListe_C=dtLinListe_C, dtLogListe_L = dtLogListe_L, dtLogListe_C=dtLogListe_C, MSDLin_Long=MSDLin_Long, MSDLin_Cour=MSDLin_Cour, stdLinLong=stdLinLong, stdLinCour=stdLinCour, MSDLog_Long=MSDLog_Long, MSDLog_Cour=MSDLog_Cour, stdLogLong=stdLogLong, stdLogCour=stdLogCour)



# ### Estimation du coefficient de diffusion


# Definition de la fonction linéaire
def lineaire(t, y):
    return t*y


# Fitting sans pondération
poptL1, _ = curve_fit(lineaire, dtLinListe_L, MSDLin_Long)
poptC1, _ = curve_fit(lineaire, dtLinListe_C, MSDLin_Cour)
poptL1, _ = curve_fit(lineaire, dtLogListe_L, MSDLog_Long)
poptC1, _ = curve_fit(lineaire, dtLogListe_C, MSDLog_Cour)
mseL1 = mean_squared_error(MSDLin_Long, lineaire(dtLinListe_L, *poptL1))  
mseC1 = mean_squared_error(MSDLin_Cour, lineaire(dtLinListe_C, *poptC1))
print("Pour le calcul des fit de courbes linéaires de MSD en fonction de dt, on obtient les MSE suivants pour une longue trajectoire vs M petites trajectoires")
print("MSE(trajectoire longue) sans pondération = ", mseL1)
print("MSE(trajectoires courtes) sans pondération = ", mseC1)

fig, ax = plt.subplots(2)             
ax[0].plot(dtLinListe_L, lineaire(dtLinListe_L, *poptL1), ls="--", color="dodgerblue")
ax[0].plot(dtLinListe_L, MSDLin_Long, 'o', color="navy")
ax[0].set_title(f"Marche aléatoire selon une loi de Bernoulli de paramètre p={p} = positions trajectoire longue", fontsize=16)
ax[0].set_xlabel("Position", fontsize=22)
ax[0].set_ylabel("Temps", fontsize=22)
ax[1].plot(dtLogListe_L, lineaire(dtLogListe_L, *poptL1), ls="--", color="dodgerblue")
ax[0].plot(dtLogListe_L, MSDLog_Long)
mpl.rcParams['axes.labelsize']=22
mpl.rcParams['legend.fontsize']=22
mpl.rcParams['axes.titlesize']=22
mpl.rcParams['figure.figsize']=(12,12)
plt.tick_params(axis='both', labelsize=22)
plt.show()  



# # Fitting avec pondération en 1/dt²
# sigmaL = dtLinListe_L
# sigmaC = dtLinListe_C
# poptL2, _ = curve_fit(lineaire, dtLinListe_L, MSD_Long, sigma = sigmaL)
# poptC2, _ = curve_fit(lineaire, dtLinListe_C, MSD_Cour, sigma = sigmaC)
# mseL2 = mean_squared_error(MSD_Long, lineaire(dtLinListe_L, *poptL2))  
# mseC2 = mean_squared_error(MSD_Cour, lineaire(dtLinListe_C, *poptC2))
# print("MSE(trajectoire longue avec pondération) = ", mseL2)
# print("MSE(trajectoires courtes avec pondération) = ", mseC2)

# #Fitting avec pondération linéaire
# # indices i = 0, 1, ..., N-1
# iL = np.arange(0, len(dtLinListe_L), 1)
# iC = np.arange(0, len(dtLinListe_C), 1)
# # poids
# wL = ((N*M) - dtLinListe_L[iL]) / ((N*M)*((N*M)-1)/2)
# wC = (N - dtLinListe_C[iC]) / (N*(N-1)/2)
# # conversion en sigma
# sigmaLinL = 1 / np.sqrt(wL)
# sigmaLinC = 1 / np.sqrt(wC)
# poptL3, _ = curve_fit(lineaire, dtLinListe_L, MSD_Long, sigma = sigmaLinL)
# poptC3, _ = curve_fit(lineaire, dtLinListe_C, MSD_Cour, sigma = sigmaLinC)

# # Calcul du coefficient de diffusion D
# D_L1 = poptL1/2
# D_C1 = poptC1/2
# D_L2 = poptL2/2
# D_C2 = poptC2/2
# D_L3 = poptL3/2
# D_C3 = poptC3/2
# print(" ")
# print("Coefficient de diffusion cas trajectoire longue sans pondération : D_L = ", D_L1)
# print("Coefficient de diffusion cas M trajectoires courtes sans pondération : D_C = ", D_C1)
# print(" ")
# print("Coefficient de diffusion cas trajectoire longue avec pondération : D_L = ", D_L2)
# print("Coefficient de diffusion cas M trajectoires courtes avec pondération : D_C = ", D_C2)
# print(" ")
# print("Coefficient de diffusion théoriques : D_th = 0.5")


# ### Tracés

# # Tracé des MSD avec pondération quadratique
# fig, ax = plt.subplots(2,2)
# fig.suptitle(f"Comparaison des MSD longue trajectoire vs M petites trajectoires pour différents dt. N={N} et M={M}\n Pondération en 1/dt²", fontsize=16)
# ax[0,0].plot(dtLinListe_L, MSD_Long, 'o',color = "blue")
# ax[1,0].plot(dtLinListe_C, MSD_Cour, 'o',color = "red")
# ax[0,0].set_xlabel("log(tau)", fontsize=14)
# ax[0,0].set_ylabel("log(MSD)", fontsize=14)
# ax[0,0].set_title(f"MSD de la longue trajectoire en fonction de tau\n D_L = {D_L1} sans pondération", fontsize=14)
# ax[1,0].set_xlabel("log(tau)", fontsize=14)
# ax[1,0].set_ylabel("log(MSD)", fontsize=14)
# ax[1,0].set_title(f"MSD des courtes trajectoires en fonction de tau\n D_C = {D_C1} sans pondération", fontsize=14)
# ax[0,0].tick_params(axis='both', labelsize=12)
# ax[1,0].tick_params(axis='both', labelsize=12)
# yFitL = poptL1*dtLinListe_L + poptL1
# ax[0,0].plot(dtLinListe_L, yFitL, label="Courbe fittée trajectoire longue", color="cyan",ls='--',lw=2)
# ax[1,0].plot(dtLinListe_C, lineaire(dtLinListe_C, *poptC1), label="Courbe fittée trajectoires courtes", color="darkorange",ls='--',lw=2)

# ax[0,1].plot(dtLinListe_L, MSD_Long, 'o',color = "blue")
# ax[1,1].plot(dtLinListe_C, MSD_Cour, 'o',color = "red")
# ax[0,1].set_xlabel("log(tau)", fontsize=14)
# ax[0,1].set_ylabel("log(MSD)", fontsize=14)
# ax[0,1].set_title(f"MSD de la longue trajectoire en fonction de tau\n D_L = {D_L2} avec pondération", fontsize=14)
# ax[1,1].set_xlabel("log(tau)", fontsize=14)
# ax[1,1].set_ylabel("log(MSD)", fontsize=14)
# ax[1,1].set_title(f"MSD des courtes trajectoires en fonction de tau\n D_C = {D_C2} avec pondération", fontsize=14)
# ax[0,1].tick_params(axis='both', labelsize=12)
# ax[1,1].tick_params(axis='both', labelsize=12)
# yFitC = poptL2[1]*dtLinListe_L + poptL2[0]
# ax[0,1].plot(dtLinListe_L, yFitC, label="Courbe fittée trajectoire longue", color="cyan",ls='--',lw=2)
# ax[1,1].plot(dtLinListe_C, lineaire(dtLinListe_C, *poptC2), label="Courbe fittée trajectoires courtes", color="darkorange",ls='--',lw=2)

# ax[0,0].set_xscale("log")
# ax[0,0].set_yscale("log")
# ax[1,0].set_xscale("log")
# ax[1,0].set_yscale("log")
# ax[0,1].set_xscale("log")
# ax[0,1].set_yscale("log")
# ax[1,1].set_xscale("log")
# ax[1,1].set_yscale("log")
# mpl.rcParams['axes.labelsize']=14

# fig.tight_layout(rect=[0, 0, 1, 0.95])
# fig.subplots_adjust(wspace=0.4, hspace = 0.5)


# #Tracé pour la pondération linéaire
# fig, ax = plt.subplots(2) 
# ax[0].plot(dtLinListe_L, MSD_Long, 'o',color = "blue")
# ax[0].set_xlabel("log(tau)", fontsize=14)
# ax[0].set_ylabel("log(MSD)", fontsize=14)
# ax[0].set_title(f"MSD de la longue trajectoire en fonction de tau\n D_L = {D_L3}\n avec pondération linéaire en 2(N-i)/((N-1)N)", fontsize=14)
# ax[0].set_xscale("log")
# ax[0].set_yscale("log")
# yFit = poptL3[1]*dtLinListe_L + poptL3[0]
# ax[0].plot(dtLinListe_L, yFit, label="Courbe fittée trajectoire longue", color="cyan",ls='--',lw=2)
# ax[1].plot(dtLinListe_C, MSD_Cour, 'o',color = "blue")
# ax[1].set_xlabel("log(tau)", fontsize=14)
# ax[1].set_ylabel("log(MSD)", fontsize=14)
# ax[1].set_title(f"MSD des courtes trajectoires en fonction de tau\n D_C = {D_C3}\n avec pondération linéaire en 2(N-i)/((N-1)N)", fontsize=14)
# ax[1].set_xscale("log")
# ax[1].set_yscale("log")
# yFit = poptC3[1]*dtLinListe_C + poptC3[0]
# ax[1].plot(dtLinListe_C, yFit, label="Courbe fittée trajectoires courtes", color="cyan",ls='--',lw=2)
# ax[0].tick_params(axis='both', labelsize=12)
# ax[1].tick_params(axis='both', labelsize=12)
# fig.tight_layout(rect=[0, 0, 1, 0.95])
# fig.subplots_adjust(hspace = 0.5)


# # Tracé de la marche aléatoire
# fig, ax = plt.subplots()             
# ax.plot(positionsLongue, [i for i in range(N*M)])  
# for k in range(M):
#     ax.plot(Mpositions[k], [i for i in range(N)], alpha=0.5)
# ax.set_title(f"Marche aléatoire selon une loi de Bernoulli de paramètre p={p} = positions trajectoire longue", fontsize=16)
# ax.set_xlabel("Position", fontsize=14)
# ax.set_ylabel("Temps", fontsize=14)
# mpl.rcParams['axes.labelsize']=14
# mpl.rcParams['legend.fontsize']=14
# mpl.rcParams['axes.titlesize']=14
# mpl.rcParams['figure.figsize']=(12,12)
# plt.tick_params(axis='both', labelsize=14)
# plt.show()                           
