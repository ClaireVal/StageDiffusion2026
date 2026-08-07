import numpy.random as rd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configuration des paramètres de visualisation
mpl.rcParams['axes.labelsize'] = 35
mpl.rcParams['legend.fontsize'] = 30
mpl.rcParams['axes.titlesize'] = 35
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'dejavusans'
mpl.rcParams['font.family'] = 'DejaVu Sans'

# Choix d'un p fixe
p = 0.5
# Initialisation des pas
N = 1000
M = 1000

# Marche 1D d'une trajectoire de N pas réguliers
positionsLongue = np.zeros(N*M)
positionsLongue[0] = 0
x = 0

for i in range(1, N*M):
    u = rd.random()
    if u < p:
        positionsLongue[i] = x - 1
        x -= 1
    else:
        positionsLongue[i] = x + 1
        x += 1

# Marche de M trajectoires de N pas réguliers
Mpositions = []
for k in range(M):
    positions = np.zeros(N)
    positions[0] = 0
    x = 0
    for i in range(1, N):
        u = rd.random()
        if u < p:
            positions[i] = x - 1
            x -= 1
        else:
            positions[i] = x + 1
            x += 1
    Mpositions.append(np.copy(positions))

### Calcul du MSD ###
def MSDLongue(x, dt):
    n = N - dt
    sum1 = 0
    for i in range(n):
        sum1 += (x[i + dt] - x[i]) ** 2
    return sum1 / n

def MSDCourtes(x, dt, M):
    sum1 = 0
    for i in range(M):
        sum1 += (x[i][dt] - x[i][0]) ** 2
    return sum1 / M


#Fonction permettant de renvoyer le fit avec les incertitudes liées au fit
def fit_lineaire(x,y):
    
    S1 = float(len(y))
    Sx = np.sum(x)
    Sxx = np.sum(x**2)
    Sxy = np.sum(x*y)
    Sy = np.sum(y)
    
    delta = S1*Sxx-Sx**2
    
    b = ( Sy*Sxx-Sx*Sxy ) / delta 
    a = ( S1*Sxy - Sx*Sy ) / delta
    
    d = a*x + b - y 
    
    sigma_y = np.sqrt( 1/(S1-2.0)*np.sum(d**2))
    
    sigma_a = np.sqrt(S1*sigma_y**2/delta)
    
    sigma_b = np.sqrt(Sxx*sigma_y**2/delta)
    
    return [a,sigma_a,b,sigma_b]


# Définition des différents pas de temps
dtLinListe = np.int_(np.linspace(0, N // 10, 100))
dtLogListe = np.unique(np.round(np.logspace(0, np.log10(N // 10), 200)).astype(int))

MSDLin_Long = np.zeros(len(dtLinListe))
MSDLin_Cour = np.zeros(len(dtLinListe))
MSDLog_Long = np.zeros(len(dtLogListe))
MSDLog_Cour = np.zeros(len(dtLogListe))

# Calcul des MSD correspondants
for ind, l in enumerate(dtLinListe):
    MSDLin_Long[ind] = MSDLongue(positionsLongue, l)
for ind, c in enumerate(dtLinListe):
    MSDLin_Cour[ind] = MSDCourtes(Mpositions, c, M)
for ind, l in enumerate(dtLogListe):
    MSDLog_Long[ind] = MSDLongue(positionsLongue, l)
for ind, c in enumerate(dtLogListe):
    MSDLog_Cour[ind] = MSDCourtes(Mpositions, c, M)

### Estimation du coefficient de diffusion ###
# Définition de la fonction affine (pour inclure l'ordonnée à l'origine)
def affine(t, a, b):
    return a + b * t

# Fitting avec calcul des incertitudes

#Cas linéaire :
#Traj unique
plt.figure()
aLin1, sigma_aLin1, bLin1, sigma_bLin1 = fit_lineaire(dtLinListe, MSDLin_Long)
print("Pour le cas linéaire et à trajectoire unique : a=",aLin1,"sigma_a=", sigma_aLin1, "b=", bLin1, "sigma_b=", sigma_bLin1)
plt.plot(dtLinListe, MSDLin_Long, "o", color="blue")
plt.plot(dtLinListe, aLin1*dtLinListe + bLin1, "--", color="red")
plt.title("Tracé du MSD et du fit dans le cas d'une trajectoire unique en échelle linéaire", fontsize=11)
plt.show()

#Traj multiples
plt.figure()
aLin2, sigma_aLin2, bLin2, sigma_bLin2 = fit_lineaire(dtLinListe, MSDLin_Long)
print("Pour le cas linéaire et à trajectoires multiples : a=",aLin2,"sigma_a=", sigma_aLin2, "b=", bLin2, "sigma_b=", sigma_bLin2)
plt.plot(dtLinListe, MSDLin_Long, "o", color="blue")
plt.plot(dtLinListe, aLin2*dtLinListe + bLin2, "--", color="red")
plt.title("Tracé du MSD et du fit dans le cas de trajectoires multiples en échelle linéaire", fontsize=11)
plt.show()


#Cas log :
#Traj unique
plt.figure()
aLog1, sigma_aLog1, bLog1, sigma_bLog1 = fit_lineaire(dtLinListe, MSDLin_Long)
print("Pour le cas linéaire et à trajectoire unique : a=",aLog1,"sigma_a=", sigma_aLog1, "b=", bLog1, "sigma_b=", sigma_bLog1)
plt.plot(dtLinListe, MSDLin_Long, "o", color="blue")
plt.plot(dtLinListe, aLog1*dtLinListe + bLog1, "--", color="red")
plt.xscale("log")
plt.yscale("log")
plt.title("Tracé du MSD et du fit dans le cas d'une trajectoire unique en échelle log", fontsize=11)
plt.show()

#Traj multiples
plt.figure()
aLog2, sigma_aLog2, bLog2, sigma_bLog2 = fit_lineaire(dtLinListe, MSDLin_Long)
print("Pour le cas linéaire et à trajectoires multiples : a=",aLog2,"sigma_a=", sigma_aLog2, "b=", bLog2, "sigma_b=", sigma_bLog2)
plt.plot(dtLinListe, MSDLin_Long, "o", color="blue")
plt.plot(dtLinListe, aLog2*dtLinListe + bLog2, "--", color="red")
plt.xscale("log")
plt.yscale("log")
plt.title("Tracé du MSD et du fit dans le cas de trajectoires multiples en échelle log", fontsize=11)
plt.show()

