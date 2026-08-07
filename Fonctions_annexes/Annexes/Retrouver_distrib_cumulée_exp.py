##Retrouver les paramètres de la loi normale à partir de la fonction cumulée

import numpy as np
import numpy.random as rd
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib as mpl


N=100
#Définition des paramètres de la loi uniforme visée
aLoi=1
bLoi = "pas défini"

### On calcule N variables aléatoires suivant la loi exponentielle de paramètre aLoi
y = np.zeros(N)
for i in range(N):
    y[i]= rd.exponential(1/aLoi)


#La fonction cumulée
res = stats.ecdf(y)                 #fonction de scipy calculant automatiquement la fonction cumulée par escaliers
X = np.array(res.cdf.quantiles)
Y = np.array(res.cdf.probabilities)

# Definition de la fonction linéaire
def lineaire(t, x, y):
    return x + t*y
# Definition de la fonction normale
def normale(x, a, b):
    return stats.norm.cdf(x, loc=a, scale=b)
# Definition de la fonction exponentielle
def exponentielle(x, a):
    return stats.expon.cdf(x, scale=1/a)

# Fitting de la courbe de F selon les trois lois candidates
popt1, pcov1 = curve_fit(lineaire, X, Y)
popt2, pcov2 = curve_fit(normale, X, Y)
popt3, pcov3 = curve_fit(exponentielle, X, Y)

# On choisit la fonction qui matche le mieux (ayant donc le plus petit mean square error MSE)
popt = []
pcov = []
nFonc = 0

#calcul des mse
mseLin = mean_squared_error(Y, lineaire(X, *popt1))
mseNorm = mean_squared_error(Y, normale(X, *popt2))
mseExp = mean_squared_error(Y, exponentielle(X, *popt3))
print("mseLin, Norm, Exp", mseLin, mseNorm, mseExp)
#choix du plus petit
if mseLin<=mseNorm:
    if mseNorm<=mseExp:
        popt, pcov = popt1, pcov1
        nFonc = 1
        
    else:
        if mseExp>=mseLin:
            popt, pcov = popt1, pcov1
            nFonc = 1
        else:
            popt, pcov = popt3, pcov3
            nFonc = 3
else:
    if mseNorm>=mseExp : 
        popt, pcov = popt3, pcov3
        nFonc = 3
    else:
        popt, pcov = popt2, pcov2
        nFonc = 2


# On définit les abscisses pour la courbe du modèle
ySort = np.sort(y)
x = np.linspace(ySort[0]-1, ySort[N-1]+1, N)

# Représentation des data originelles et de la courbe fittée
fig, ax = plt.subplots()  
ax.scatter(X, Y, label="Données", color="blue")
if nFonc==1:
    x0 = popt[0]
    y0 = popt[1]
    aFit = -x0/y0        #On peut directement relier x et y aux valeurs de la loi uniforme en explicitant l'intégrale F(t) en fonction de a et b
    bFit = (1-x0)/y0
    ax.plot(x, lineaire(x, *popt), label="Courbe fittée", color="red")
    print("Loi uniforme. Paramètres fittés:", aFit, "et", bFit, "et paramètres de la loi (aLoi,bLoi):", aLoi, "et", bLoi)
if nFonc==2:
    ax.plot(x, normale(x, *popt), label="Courbe fittée", color="red")
    print("Loi normale. Paramètres fittés:", popt[0], "et", popt[1], "et paramètres de la loi (aLoi, bLoi):", aLoi, " et ", bLoi)
if nFonc==3:
    ax.plot(x, exponentielle(x, *popt), label="Courbe fittée", color="red")
    print("Loi exponentielle. Paramètre fitté:", popt[0], "et paramètre de la loi (aLoi):", aLoi)
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
