import numpy.random as rd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

#mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#mpl.rc('text', usetex=True)
mpl.rcParams['ytick.labelsize']=30
mpl.rcParams['xtick.labelsize']=30
#mpl.rcParams['text.fontsize']=20
mpl.rcParams['axes.labelsize']=35
mpl.rcParams['legend.fontsize']=30
mpl.rcParams['axes.titlesize']=35
mpl.rcParams['figure.figsize']=(12,12)

# Choix d'un p fixe
p = 0.5
# Initialisation des pas
N= 100
M = 100



# Marche 1D d'une trajectoire de M*N pas réguliers

positionsLongue = []
positionsLongue.append(0)
x = 0

for i in range(M*N):
    u = rd.random()
    if u<p:
        positionsLongue+=[x-1]
        x-=1
    else:
        positionsLongue+=[x+1]
        x+=1

# Marche de M trajectoires de N pas réguliers

Mpositions = []

for k in range(M):
    positions = []
    positions.append(0)
    x = 0
    for i in range(N):
        u = rd.random()
        if u<p:
            positions+=[x-1]
            x-=1
        else:
            positions+=[x+1]
            x+=1
    Mpositions.append(positions.copy())




###  Calcul du MSD  ###

# Premier cas : Moyenne sur une grande trajectoire de M*N points
def MSDLongue(x,dt):
    n = M*N - dt                         #nombre de (t1,t2) tels que t2-t1=dt
    sum = 0
    for i in range(n):
        sum += (x[i+dt]-x[i])**2
    return sum/n

# Second cas : Moyenne sur M trajectoires (à parallèliser)
def MSDCourtes(x, dt, M):
    #x a la dimension M*N
    n = N-dt                          #nb de pas de temps à considérer
    sum = 0
    for k in range(n):                          #on parcourt les M trajectoires à t fixé
        for i in range(M):                      #on parcourt les M trajectoire
            sum += (x[i][k+dt]-x[i][k])**2       #on considère les x de la k-ième trajectoire, et au i-ième pas de temps
    return sum/(M*n)

# Définition des différents pas de temps
dtLinListe_L = np.int_(np.linspace(10, (M*N)//10, 1000))
dtLinListe_C =  np.int_(np.linspace(10, N//10, 100))
    
MSD_Long = []
MSD_Cour = []

# Calcul des MSD correspondants
for l in dtLinListe_L:
    MSD_Long.append(MSDLongue(positionsLongue, l))
for c in dtLinListe_C:
    MSD_Cour.append(MSDCourtes(Mpositions,c, M))
    

# Calcul de l'écart-type
positionsL = np.square(positionsLongue)
positionsC = np.square(Mpositions)

sigmaL = np.std(positionsLongue)
sigmaC = np.std(np.array(Mpositions).flatten())

print("les écarts-types sont pour la longue :", sigmaL, "et pour les courtes :", sigmaC)



### Estimation du coefficient de diffusion
# Definition de la fonction linéaire
def lineaire(t, x, y):
    return x + t*y

# Fitting
poptL, _ = curve_fit(lineaire, dtLinListe_L, MSD_Long)
poptC, _ = curve_fit(lineaire, dtLinListe_C, MSD_Cour)
mseL = mean_squared_error(MSD_Long, lineaire(dtLinListe_L, *poptL))  
mseC = mean_squared_error(MSD_Cour, lineaire(dtLinListe_C, *poptC))
print("Pour le calcul des fit de courbes linéaires de MSD en fonction de dt, on obtient les MSE suivants pour une longue trajectoire vs M petites trajectoires")
print("MSE(trajectoire longue) = ", mseL)
print("MSE(trajectoires courtes) = ", mseC)

# Calcul du coefficient de diffusion D
D_L = poptL[1]/2
D_C = poptC[1]/2
print(" ")
print("Coefficient de diffusion cas trajectoire longue : D_L = ", D_L)
print("Coefficient de diffusion cas M trajectoires courtes : D_C = ", D_C)
print("Coefficient de diffusion théoriques : D_th = 0.5")

# Boucle de test
borneMin = [5,10]                           #borne min
borneMax = [N//10, N//20, N//50]            #borne max
resultats = {}

for bmin in borneMin:
    for bmax in borneMax:
        
        key = (bmin, bmax)
        resultats[key] = []
        
        for _ in range(5):                  #nb répétitions
            
            # génération du linspace
            paramsL = np.linspace(bmin, bmax, 1000, dtype=int)
            paramsC = np.linspace(bmin, bmax, 100, dtype=int)
            
            MSD_courant_L = []
            MSD_courant_C = []
            
            for l in paramsL:
                val = MSDLongue(positionsLongue, l)
                MSD_courant_L.append(val)
            
            for l in paramsC:
                val = MSDCourtes(Mpositions, l, M)
                MSD_courant_C.append(val)
            
            MSD_courant_L = np.array(MSD_courant_L)
            MSD_courant_C = np.array(MSD_courant_C)
        
            poptTestL, _ = curve_fit(lineaire, paramsL, MSD_courant_L)
            poptTestC, _ = curve_fit(lineaire, paramsC, MSD_courant_C)
            resultats[key].append((poptTestL[1] / 2, poptTestC[1]/2))

for key, valeurs in resultats.items():
    valeurs = np.array(valeurs)
    
    print(f"Params {key}:")
    print("  moyenne =", np.mean(valeurs))
    print("  std     =", np.std(valeurs))
      
    
# Tracé des MSD

fig, ax = plt.subplots(2)
fig.suptitle(f"Comparaison des MSD longue trajectoire vs M petites trajectoires pour différents dt. N={N} et M={M}", fontsize=16)
ax[0].plot(dtLinListe_L, MSD_Long, color = "blue")
ax[1].plot(dtLinListe_C, MSD_Cour, color = "red")
ax[0].set_xlabel("tau", fontsize=14)
ax[0].set_ylabel("MSD", fontsize=14)
ax[0].set_title(f"MSD de la longue trajectoire en fonction de tau, D_L = {D_L}", fontsize=14)
ax[1].set_xlabel("tau", fontsize=14)
ax[1].set_ylabel("MSD", fontsize=14)
ax[1].set_title(f"MSD des courtes trajectoires en fonction de tau, D_C = {D_C}", fontsize=14)
ax[0].tick_params(axis='both', labelsize=12)
ax[1].tick_params(axis='both', labelsize=12)
ax[0].plot(dtLinListe_L, lineaire(dtLinListe_L, *poptL), label="Courbe fittée trajectoire longue", color="cyan")
ax[1].plot(dtLinListe_C, lineaire(dtLinListe_C, *poptC), label="Courbe fittée trajectoires courtes", color="magenta")
mpl.rcParams['axes.labelsize']=14



# Tracé de la marche aléatoire

fig, ax = plt.subplots()             
ax.plot(positionsLongue, [i for i in range((N*M)+1)])  
ax.set_title(f"Marche aléatoire selon une loi de Bernoulli de paramètre p={p} = positions trajectoire longue", fontsize=16)
ax.set_xlabel("Position", fontsize=14)
ax.set_ylabel("Temps", fontsize=14)
mpl.rcParams['axes.labelsize']=14
mpl.rcParams['legend.fontsize']=14
mpl.rcParams['axes.titlesize']=14
mpl.rcParams['figure.figsize']=(12,12)
plt.tick_params(axis='both', labelsize=14)
plt.show()                           
