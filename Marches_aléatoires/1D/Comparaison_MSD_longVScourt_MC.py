import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

mpl.rcParams['axes.labelsize'] = 35
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['axes.titlesize'] = 35
mpl.rcParams['figure.figsize'] = (12, 12)

mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'dejavusans'
mpl.rcParams['font.family'] = 'DejaVu Sans'

# Paramètres de la simulation
p=0.5          # probabilité de partir à gauche
N=5000        # longueur d'une trajectoire
M=5000        # nombre de trajectoires courtes

# nombre de réalisations Monte-Carlo
N_MC=2000


def affine(t, a, b):
    return a + b*t

# Génération d'une marche aléatoire de longueur N
def genere_marche(N, p):
    positions = np.zeros(N)
    x = 0
    for i in range(1, N):
        if rd.random() < p:
            x -= 1
        else:
            x += 1
        positions[i] = x
    return positions


# Génération de M marches
def genere_M_marches(M, N, p):
    marches = np.zeros((M, N))
    for k in range(M):
        marches[k] = genere_marche(N, p)
    return marches


# MSD d'une longue trajectoire
def MSDLongue(positions, dt):
    return np.mean((positions[dt:] - positions[:-dt])**2)


# MSD des trajectoires courtes
def MSDCourtes(marches, dt):
    return np.mean((marches[:, dt] - marches[:, 0])**2)

# Calcul complet des MSD
def calcule_MSD(position_longue, positions_courtes, dtLinListe, dtLogListe):
    MSDLin_Long = np.zeros(len(dtLinListe))
    MSDLin_Cour = np.zeros(len(dtLinListe))
    MSDLog_Long = np.zeros(len(dtLogListe))
    MSDLog_Cour = np.zeros(len(dtLogListe))

    for i,dt in enumerate(dtLinListe):
        MSDLin_Long[i] = MSDLongue(position_longue,dt)
        MSDLin_Cour[i] = MSDCourtes(positions_courtes,dt)

    for i,dt in enumerate(dtLogListe):
        MSDLog_Long[i] = MSDLongue(position_longue,dt)
        MSDLog_Cour[i] = MSDCourtes(positions_courtes,dt)

    return (MSDLin_Long, MSDLin_Cour, MSDLog_Long, MSDLog_Cour)


# Ajustement linéaire
def fit_MSD(dt, MSD):

    popt, pcov = curve_fit(affine, dt, MSD)
    pente = popt[1]
    ordonnee = popt[0]
    sigma = np.sqrt(np.diag(pcov))

    sigma_a = sigma[0]
    sigma_b = sigma[1]
    D = pente/2
    sigma_D = sigma_b/2

    return {"popt": popt, "pcov": pcov,"a": ordonnee,"b": pente,"sigma_a": sigma_a, "sigma_b": sigma_b,"D": D,"sigma_D": sigma_D}


# Listes des temps
dtLinListe = np.int_(np.linspace(1, N//10, 100))

dtLogListe = np.unique(np.round(np.logspace(0, np.log10(N//10),200)).astype(int))


###########################   Première simulation
position_longue = genere_marche(N,p)
positions_courtes = genere_M_marches(M,N,p)

(MSDLin_Long, MSDLin_Cour, MSDLog_Long, MSDLog_Cour) = calcule_MSD(position_longue, positions_courtes,dtLinListe,dtLogListe)


# Premier fit

fit_L1 = fit_MSD(dtLinListe,MSDLin_Long)
fit_C1 = fit_MSD(dtLinListe,MSDLin_Cour)

fit_L2 = fit_MSD(dtLogListe,MSDLog_Long)
fit_C2 = fit_MSD(dtLogListe,MSDLog_Cour)


# Génération rapide (vectorisée) de marches aléatoires

def genere_marche_rapide(N, p):
    pas = np.where(rd.random(N-1) < p, -1, 1)
    position = np.zeros(N)
    position[1:] = np.cumsum(pas)
    return position

def genere_M_marches_rapide(M, N, p):
    pas = np.where(rd.random((M, N-1)) < p, -1, 1)
    marches = np.zeros((M, N))
    marches[:,1:] = np.cumsum(pas, axis=1)
    return marches



#########################    Boucle Monte Carlo

D_Lin_Long_MC = np.zeros(N_MC)
D_Lin_Court_MC = np.zeros(N_MC)

D_Log_Long_MC = np.zeros(N_MC)
D_Log_Court_MC = np.zeros(N_MC)


for mc in range(N_MC):
    if mc % 50 == 0:
        print("Réalisation Monte Carlo :", mc, "/", N_MC)

    # Nouvelle réalisation
    position_longue_MC = genere_marche_rapide(N,p)
    positions_courtes_MC = genere_M_marches_rapide(M,N,p)
    # Calcul MSD
    (MSDLin_Long_MC, MSDLin_Cour_MC, MSDLog_Long_MC, MSDLog_Cour_MC) = calcule_MSD(position_longue_MC, positions_courtes_MC, dtLinListe, dtLogListe)

    # Fits
    fit_L1_MC = fit_MSD(dtLinListe, MSDLin_Long_MC)
    fit_C1_MC = fit_MSD(dtLinListe, MSDLin_Cour_MC)

    fit_L2_MC = fit_MSD(dtLogListe, MSDLog_Long_MC)
    fit_C2_MC = fit_MSD(dtLogListe, MSDLog_Cour_MC)


    # Stockage des coefficients D
    D_Lin_Long_MC[mc] = fit_L1_MC["D"]
    D_Lin_Court_MC[mc] = fit_C1_MC["D"]
    D_Log_Long_MC[mc] = fit_L2_MC["D"]
    D_Log_Court_MC[mc] = fit_C2_MC["D"]


#######################    Résultats Monte Carlo

D_results = {"Longue trajectoire linéaire":2*D_Lin_Long_MC,"Courtes trajectoires linéaires":2*D_Lin_Court_MC,"Longue trajectoire logarithmique":2*D_Log_Long_MC,"Courtes trajectoires logarithmiques":2*D_Log_Court_MC}


print("\n================================")
print("RESULTATS MONTE CARLO")
print("================================")


for nom,D_values in D_results.items():
    moyenne = np.mean(D_values)
    ecart_type = np.std(D_values, ddof=1)
    print(f"{nom} : "f"D = {moyenne:.5f} ± {ecart_type:.5f}")
    


######################    Tracés finaux et histogrammes Monte Carlo

# Calcul des moyennes et écarts-types Monte Carlo
resultats_MC = {}
for nom, valeurs in D_results.items():
    resultats_MC[nom] = (2*np.mean(valeurs),np.std(valeurs, ddof=1))

# Affichage des coefficients de diffusion

print("\n================================")
print("COEFFICIENTS DE DIFFUSION MONTE CARLO")
print("================================")

for nom, (moy, sigma) in resultats_MC.items():
    print(f"{nom} : D = {moy:.5f} ± {sigma:.5f}")


# Tracé des MSD avec les fits

fig, ax = plt.subplots()

# Longue trajectoire linéaire
ax.plot(dtLinListe, MSDLin_Long, 'o', markersize=8, color="slateblue", label="MSD longue trajectoire")
ax.plot(dtLinListe, affine(dtLinListe, fit_L1["a"],fit_L1["b"]), '--', linewidth=3, color="navy", label=f"Fit long : pente={resultats_MC['Longue trajectoire linéaire'][0]:.3f}")

# Courtes trajectoires
ax.plot(dtLinListe, MSDLin_Cour, '^', markersize=8, color="green", label="MSD courtes trajectoires")
ax.plot(dtLinListe, affine(dtLinListe,fit_C1["a"], fit_C1["b"]),'--', linewidth=3, color="darkgreen", label=f"Fit court : pente={resultats_MC['Courtes trajectoires linéaires'][0]:.3f}")

ax.set_title(f"Marche aléatoire 1D (p={p})\n" f"Estimation Monte Carlo de D", fontsize=16)
ax.set_xlabel("Temps dt", fontsize=22)
ax.set_ylabel("MSD",fontsize=22)
ax.tick_params(axis='both', labelsize=18)
ax.legend()
plt.tight_layout()
plt.show()

# Histogrammes des coefficients D

fig, axes = plt.subplots(2, 2, figsize=(14,12))
axes = axes.flatten()

for ax, (nom, valeurs) in zip(axes,D_results.items()):

    moyenne = np.mean(valeurs)
    sigma = np.std(valeurs, ddof=1)
    ax.hist(valeurs, bins=40)
    ax.axvline(moyenne,linestyle='--', linewidth=2, label=f"moyenne={moyenne:.3f}")
    ax.axvline(moyenne-sigma, linestyle=':', linewidth=2)
    ax.axvline(moyenne+sigma, linestyle=':', linewidth=2)
    ax.set_title(nom,fontsize=14)
    ax.set_xlabel("pente", fontsize=14)
    ax.set_ylabel("Nombre", fontsize=14)
    ax.legend()

plt.tight_layout()
plt.show()
