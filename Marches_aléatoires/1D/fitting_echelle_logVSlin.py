import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
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


data = np.load("MSD.npz")
dtLinListe_L = data["dtLinListe_L"]
dtLinListe_C=data["dtLinListe_C"]
dtLogListe_L = data["dtLogListe_L"]
dtLogListe_C=data["dtLogListe_C"]
MSDLin_Long=data["MSDLin_Long"]
MSDLin_Cour=data["MSDLin_Cour"]
MSDLog_Long=data["MSDLog_Long"]
MSDLog_Cour=data["MSDLog_Cour"]
stdLinLong=data["stdLinLong"]
stdLinCour=data["stdLinCour"]
stdLogLong=data["stdLogLong"]
stdLogCour=data["stdLogCour"]

N= 1000
M = 100

# Definition des fonctions linéaire et affine
def affine(t, y):
    return t*y
# Definition de la fonction linéaire
def lineaire(t, x, y):
    return x + t*y

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


#######################################################
################ CAS LINEAIRE #########################
#######################################################

print("---------  ECHELLE LINEAIRE  ---------")

### Estimation du coefficient de diffusion

# Fitting sans pondération
poptL1,_ = curve_fit(affine, dtLinListe_L, MSDLin_Long)
poptC1,_ = curve_fit(affine, dtLinListe_C, MSDLin_Cour)
# mseL1 = mean_squared_error(MSDLin_Long, affine(dtLinListe_L, poptL1))  
# mseC1 = mean_squared_error(MSDLin_Cour, affine(dtLinListe_C, poptC1))
# print("Pour le calcul des fit de courbes linéaires de MSD en fonction de dt, on obtient les MSE suivants pour une longue trajectoire vs M petites trajectoires")
# print("MSE(trajectoire longue) sans pondération = ", mseL1)
# print("MSE(trajectoires courtes) sans pondération = ", mseC1)

# Fitting avec pondération en 1/dt²
sigmaL = dtLinListe_L
sigmaC = dtLinListe_C
sigmaL[sigmaL == 0] = 1e-8                                                               #==> On les remplace par une petite valeur non nulle
sigmaC[sigmaC == 0] = 1e-8
poptL2,_  = curve_fit(affine, dtLinListe_L, MSDLin_Long, sigma = sigmaL)
poptC2,_  = curve_fit(affine, dtLinListe_C, MSDLin_Cour, sigma = sigmaC)
# mseL2 = mean_squared_error(MSDLin_Long, affine(dtLinListe_L, poptL2))  
# mseC2 = mean_squared_error(MSDLin_Cour, affine(dtLinListe_C, poptC2))
# print("MSE(trajectoire longue avec pondération) = ", mseL2)
# print("MSE(trajectoires courtes avec pondération) = ", mseC2)

#Fitting avec pondération linéaire
# indices i = 0, 1, ..., N-1
iL = np.arange(0, len(dtLinListe_L), 1)
iC = np.arange(0, len(dtLinListe_C), 1)
# poids
wL = ((N*M) - dtLinListe_L[iL]) / ((N*M)*((N*M)-1)/2)
wC = (N - dtLinListe_C[iC]) / (N*(N-1)/2)
# conversion en sigma
sigmaLinL = 1 / np.sqrt(wL)
sigmaLinC = 1 / np.sqrt(wC)
poptL3,_  = curve_fit(affine, dtLinListe_L, MSDLin_Long, sigma = sigmaLinL)
poptC3,_  = curve_fit(affine, dtLinListe_C, MSDLin_Cour, sigma = sigmaLinC)

#Fitting en échelle linéaire avec pondération avec std
print("MIN/MAX des STD (trajectoire longue) : ", np.min(stdLinLong), np.max(stdLinLong))        #Il y a des 0 dans la liste qui faussent la pondération en donnant des poids énormes à certains points
stdLinLong[stdLinLong == 0] = 1e-8                                                              #==> On les remplace par une petite valeur non nulle
stdLinCour[stdLinCour == 0] = 1e-8

poptL4, _ = curve_fit(affine, dtLinListe_L, MSDLin_Long, sigma = stdLinLong)
poptC4, _ = curve_fit(affine, dtLinListe_C, MSDLin_Cour, sigma = stdLinCour)

# Calcul du coefficient de diffusion D
D_L1 = poptL1/2
D_C1 = poptC1/2
D_L2 = poptL2/2
D_C2 = poptC2/2
D_L3 = poptL3/2
D_C3 = poptC3/2
D_L4 = poptL4/2
D_C4 = poptC4/2
print(" ")
print(f"Coefficient de diffusion cas trajectoire longue sans pondération : D_L = {D_L1[0]:.20f}")
print(f"Coefficient de diffusion cas M trajectoires courtes sans pondération : D_C = {D_C1[0]:.20f}")
print(" ")
print(f"Coefficient de diffusion cas trajectoire longue avec pondération en 1/dt²: D_L = {D_L2[0]:.20f}")
print(f"Coefficient de diffusion cas M trajectoires courtes avec pondération en 1/dt² : D_C = {D_C2[0]:.20f}")
print(" ")
print(f"Coefficient de diffusion cas trajectoire longue avec pondération linéaire : D_L = {D_L3[0]:.20f}")
print(f"Coefficient de diffusion cas M trajectoires courtes avec pondération linéaire: D_C = {D_C3[0]:.20f}")
print(" ")
print(f"Coefficient de diffusion cas trajectoire longue avec pondération std : D_L = {D_L4[0]:.20f}")
print(f"Coefficient de diffusion cas M trajectoires courtes avec pondération std : D_C = {D_C4[0]:.20f}")
print(" ")
print("Coefficient de diffusion théoriques : D_th = 0.5")


coeff_lin = poptL1.copy()
Dlin = D_L1.copy()
### Tracés




##################################################
################ CAS LOG #########################
##################################################

print(" ")
print("---------  ECHELLE LOG  ---------")



# Transformation en log
mask = (MSDLog_Long > 0)

log_dt_L = np.log(dtLogListe_L[mask])
log_MSD_L = np.log(MSDLog_Long[mask])


log_dt_L = np.log(dtLogListe_L)
log_MSD_L = np.log(MSDLog_Long)
log_dt_C = np.log(dtLogListe_C)
log_MSD_C = np.log(MSDLog_Cour)

# Fit sans pondération
poptL1, _ = curve_fit(lineaire, log_dt_L, log_MSD_L)
poptC1, _ = curve_fit(lineaire, log_dt_C, log_MSD_C)
D_L1 = 0.5 * np.exp(poptL1[0])  # D = 0.5 * exp(ordonnée à l'origine)
D_C1 = 0.5 * np.exp(poptC1[0])

print("Coefficient de diffusion (log, sans pondération) : D_L = ", D_L1)

# Fitting  en échelle log/log avec pondération en 1/dt²
sigmaL = dtLogListe_L
sigmaC = dtLogListe_C
poptL2, _ = curve_fit(lineaire, dtLogListe_L, MSDLog_Long, sigma = sigmaL)
poptC2, _ = curve_fit(lineaire, dtLogListe_C, MSDLog_Cour, sigma = sigmaC)


#Fitting  en échelle log/log avec pondération linéaire
# indices i = 0, 1, ..., N-1
iL = np.arange(0, len(dtLogListe_L), 1)
iC = np.arange(0, len(dtLogListe_C), 1)
# poids
wL = ((N*M) - dtLogListe_L[iL]) / ((N*M)*((N*M)-1)/2)
wC = (N - dtLogListe_C[iC]) / (N*(N-1)/2)

# conversion en sigma
sigmaLogL = 1 / np.sqrt(wL)
sigmaLogC = 1 / np.sqrt(wC)
poptL3, _ = curve_fit(lineaire, dtLogListe_L, MSDLog_Long, sigma = sigmaLogL)
poptC3, _ = curve_fit(lineaire, dtLogListe_C, MSDLog_Cour, sigma = sigmaLogC)

#Fitting en échelle log/log avec pondération avec std
print("MIN/MAX des STD (trajectoire longue) : ", np.min(stdLogLong), np.max(stdLogLong))         #Il y a des 0 dans la liste qui faussent la pondération en donnant des poids énormes à certains points
stdLogLong[stdLogLong == 0] = 1e-8                                                               #==> On les remplace par une petite valeur non nulle
stdLogCour[stdLogCour == 0] = 1e-8

poptL4, _ = curve_fit(lineaire, np.log(dtLogListe_L), np.log(MSDLog_Long), sigma = stdLogLong)
poptC4, _ = curve_fit(lineaire, np.log(dtLogListe_C), np.log(MSDLog_Cour), sigma = stdLogCour)

# Calcul du coefficient de diffusion D
D_L1 = exp(poptL1[0])/2
D_C1 = exp(poptC1[0])/2
D_L2 = exp(poptL2[0])/2
D_C2 = exp(poptC2[0])/2
D_L3 = exp(poptL3[0])/2
D_C3 = exp(poptC3[0])/2
D_L4 = exp(poptL4[0])/2
D_C4 = exp(poptC4[0])/2

print("")
print("Coefficient de diffusion cas trajectoire longue sans pondération : D_L = ", D_L1)
print("Coefficient de diffusion cas M trajectoires courtes sans pondération : D_C = ", D_C1)
print(" ")
print("Coefficient de diffusion cas trajectoire longue avec pondération 1/t² : D_L = ", D_L2)
print("Coefficient de diffusion cas M trajectoires courtes avec pondération 1/t² : D_C = ", D_C2)
print(" ")
print("Coefficient de diffusion cas trajectoire longue avec pondération linéaire : D_L = ", D_L3)
print("Coefficient de diffusion cas M trajectoires courtes avec pondération linéaire: D_C = ", D_C3)
print(" ")
print("Coefficient de diffusion cas trajectoire longue avec pondération std : D_L = ", D_L4)
print("Coefficient de diffusion cas M trajectoires courtes avec pondération std : D_C = ", D_C4)
print(" ")
print("Coefficient de diffusion théoriques : D_th = 0.5")         



#Tracé lin vs log sans pondération

fig, ax = plt.subplots(2)

# fig.suptitle(f"Comparaison des MSD en échelle linéaire vs logarithmique. N={N} et M={M}", fontsize=16)
ax[0].plot(dtLinListe_L, MSDLin_Long, 'o',color = "steelblue")
ax[1].plot(dtLogListe_L, MSDLog_Long, 'o',color = "indigo")
ax[0].set_xlabel("t", fontsize=20)
ax[0].set_ylabel("MSD", fontsize=20)
ax[0].set_title(f"D = {Dlin[0]:.3f} pour une unique trajectoire longue en échelle linéaire de {N*M} pas", fontsize=20)
ax[1].set_xlabel("t", fontsize=20)
ax[1].set_ylabel("MSD", fontsize=20)
ax[1].set_title(f"D = {D_L1:.3f} pour une unique trajectoire longue en échelle logarithmique de {N*M} pas", fontsize=20)
ax[0].tick_params(axis='both', labelsize=20)
ax[1].tick_params(axis='both', labelsize=20)
ax[1].set_xscale("log")
ax[1].set_yscale("log")
yFitL = coeff_lin * dtLinListe_L
yFit2 = poptL1[1]*dtLogListe_L + poptL1[0]
ax[0].plot(dtLinListe_L, yFitL, label="Courbe fittée en échelle linéaire", color="cyan",ls='--',lw=2)
ax[1].plot(dtLogListe_L, yFit2, label="Courbe fittée en échelle logarithmique", color="slateblue",ls='--',lw=2)
fig.subplots_adjust(hspace = 0.5)
plt.savefig("logvslin.svg")

a, sigmaa, b, sigmab = fit_lineaire(dtLinListe_L, MSDLin_Long)
aL, sigmaaL, bL, sigmabL = fit_lineaire(dtLogListe_L, MSDLog_Long)

print("LIN", a, sigmaa, b, sigmab)
print("LOG", aL, sigmaaL, bL, sigmabL)
