###Retrouver les paramètres de la loi uniforme à partir de la fonction cumulée

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy.random as rd

N=100000
#Définition des paramètres de la loi uniforme visée
aLoi=1
bLoi=3

### On calcule N variables aléatoires uniformes sur [a,b]
y = np.zeros(N)
for i in range(N):
    u = rd.random()
    y[i]= aLoi + u*(bLoi-aLoi)

#La fonction cumulée
ySort = np.sort(y)
F = np.arange(1, N+1) / N  #pas de valeur 1/N car on a c'est les valeurs discrètes possibles de F(x). F[i] vaut F(ySort[i])

# Definition de la fonction linéaire
def lineaire(t, x, y):
    return x + t*y

# Fitting de la courbe de F selon une loi linéaire
popt, pcov = curve_fit(lineaire, ySort, F)
x = popt[0]
y = popt[1]
aFit = -x/y        #On peut directement relier x et y aux valeurs de la loi uniforme en explicitant l'intégrale F(t) en fonction de a et b
bFit = (1-x)/y

# On définit les abscisses pour la courbe du modèle
x = np.linspace(ySort[0]-1, ySort[N-1]+1,N)

# Représentation des data originelles et de la courbe fittée
fig, ax = plt.subplots()  
ax.scatter(ySort, F, label='Original Data', color='blue')
ax.plot(x, lineaire(x, *popt), label='Fitted Line', color='red')
plt.legend()
ax.set_title("Fonction de répartition expérimentale, et courbe fittée de la fonction", fontsize=16)
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("F(x)", fontsize=14)
mpl.rcParams['axes.labelsize']=14
mpl.rcParams['legend.fontsize']=14
mpl.rcParams['axes.titlesize']=14
mpl.rcParams['figure.figsize']=(12,12)
plt.tick_params(axis='both', labelsize=14)
plt.show()

# Displaying the optimal parameters (slope and intercept)
print("Paramètres fittés (a, b):", aFit, " ", bFit, "et paramètres de la loi (aLoi, bLoi):", aLoi, " ", bLoi)
