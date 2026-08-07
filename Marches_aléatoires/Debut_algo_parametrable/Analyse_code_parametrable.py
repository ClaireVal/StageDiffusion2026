import matplotlib.pyplot as plt
from math import exp
from ase.units import kB
import numpy as np
from numba import njit, prange
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import pickle

# Géométrie du système
X=10
Y=10
Z=10
Natomes = X*Y*Z
N=10000            #nb de pas

swap=1             #activation du swap si 1, désactivation si 0
tpsVar=1        #si =0 tps constant, si =1 temps variable

x0 = [X//2, Y//2, Z//2]


### Paramètres du système
# # L'ordre de grandeur pour les énergies de formation et de migration d'un défaut sont de l'ordre de 1eV (cf rapport de stage de Manon Dewynter)
G_M = 1.5 #eV = barrière d'énergie pour l'échange lacune / atome du réseau
T = 300 #K
nu_0 = 10e13

# Sites des plus proches voisins:
ppv = np.array([(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)], dtype='float64')
Nppv=len(ppv)
Lppv = np.linalg.norm(ppv)
dimension = ppv.shape[1]

freqAt = nu_0 * exp(-G_M/(kB*T))               #Fréquence de saut des atomes du réseau
distMaille=np.sum(ppv**2, axis=1)

aMaille = np.mean(distMaille)               ### TO-DO : Différencier selon chaque direction (x,y,z) ==> pas besoin de faire une moyenne qui représente de manière inexacte le comportement

# Caractérisation des défauts mis en jeu:
inputDef = np.array([[freqAt*1000, 0.001], [freqAt, 0.001], [freqAt*0.001, 0.001]])
nbLabel = np.shape(inputDef)[0]
if np.sum(inputDef[:, 1])>1:
    print("Concentration totale en défauts supérieure à 100%")

freqDef = inputDef[:,0]
cDef = inputDef[:,1]
alpha = freqDef/freqAt


def tmax_par_label(dicoDef):
    tmax=np.zeros((nbLabel))          #On stocke en mémoire un Ntmax par label pour reconstruire ensuite un axe d'abscisses dt par type de défauts
    for k in range(len(dicoDef)):
        lab=labelsDefauts[k]
        tmaxCourant=list(dicoDef.values())[k][1][-1][0]
        if tmaxCourant>tmax[lab]:
            tmax[lab]=tmaxCourant
    return tmax


@njit(parallel=True)
def MSD_defauts(dicoDef, nbBins=100):           #nbBins = Nb de valeurs de dt que l'on prend pour le calcul de MSD(dt) (i.e le nb d'abscisses)
    
    #On calcule, pour chaque label, la trajectoire la plus longue, donnant le dtmax pour le calcul du MSD(dt)
    tmax=tmax_par_label(dicoDef)                     
    
    #On prépare une liste de résultats, répertoriant pour chaque label (i.e fréquence de saut), le MSD associé. Comme on stocke le nombre de trajectoires ayant participé au calcul, on peut aussi envisager d'ajouter des trajectoires au calcul pour une meilleure convergence et compatibilité avec le fonctionnement des calculateurs
    MSD_par_label = np.zeros((nbLabel, nbBins+1))  #On stocke pour chaque label la valeur du nombre de trajectoires prises en compte pour la moyenne (en dernière position) et le MSD associé (nbBins valeurs)(soit NnbBins+1 valeurs au final)
    dt_centers_par_label = np.zeros((nbLabel, 2, nbBins)) #On stocke pour chaque label les valeurs des abscisses des bins(il y en a nbBins) et des centres des bins (il y en a nbBins-1)
    counts = np.zeros((nbLabel,nbBins), dtype=np.int64) #Garde en mémoire le nombre de dt par label ayant été additionnés pour obtenir MSD_par_label[label][k], pour tout label et k
    
    
    #On parcourt ensuite les valeurs du dictionnaire défauts en calculant le MSD séparemment pour chaque type de défauts
    for k in range(nbDef):
        defautCourant = list(dicoDef.values())[k][1]
        freqCourant = list(dicoDef.values())[k][0]
        labelCourant = labelsDefauts[k]
        
        #Génération ou récupération des abscisses dt de MSD(dt)ˇ
        #Si aucun MSD n'a déjà été calculé pour ce label de défauts, alors on crée une liste d'abscisses pour le MSD, qui sera la même pour tous les défauts ayant le même label
        if MSD_par_label[labelCourant,-1]==0:
            #Détermination des abscisses minimale et maximale pour les dt:              
            #Calcul de dtmin
            if freqCourant>=freqAt:
                dtmin=1/(4*freqCourant)
            else:
                dtmin=1/(3*freqAt+freqCourant)              
            #Calcul de dtmax
            dtmax=tmax[labelCourant] * 0.5
            
            log_min = np.log10(dtmin)
            log_max = np.log10(dtmax)

            # bins log uniformes
            dt_bins = np.empty(nbBins)
            step = (log_max - log_min) / (nbBins - 1)
            for i in range(nbBins):
                dt_bins[i] = 10.0 ** (log_min + i * step)

            dt_centers = np.empty(nbBins - 1)
            for i in range(nbBins - 1):
                dt_centers[i] = np.sqrt(dt_bins[i] * dt_bins[i+1])

            # facteur pour binning rapide
            inv_step = 1.0 / step
            
            dt_centers_par_label[labelCourant,0]=dt_bins                #On garde en mémoire l'axe des abscisses du MSD (les valeurs des centres et des bins des dt)
            dt_centers_par_label[labelCourant,1,:-1]=dt_centers
        
        
        else:
            #On récupère les dt_bins et dt_centers stockés dans dt_centers_par_label
            dt_bins=dt_centers_par_label[labelCourant,0]
            dt_centers=dt_centers_par_label[labelCourant,1,:-1]
            
            #On retrouve les autres grandeurs utiles à partir de ces deux listes
            log_max = np.log10(dt_bins[-1])
            log_min = np.log10(dt_bins[0])
            inv_step = (nbBins-1)/(log_max-log_min)
        
        
        max_dt = dt_bins[-1]
        NtCourant=len(defautCourant)
        
        #On parcourt tous les couples ((xi,ti),(xj,tj)) avec ti, tj appartenant aux instants de sauts du défaut k, et ti<tj
        for i in prange(NtCourant-1):
            xi = defautCourant[i][1]        
            ti = defautCourant[i][0]
            
            for j in range(i + 1, NtCourant):
                dt_ij = defautCourant[j][0] - ti
                if dt_ij > max_dt:
                    break
    
                # calcul dx sans allocation
                dx = 0.0
                for d in range(dimension):   # marche pour dim > 1
                    tmp = defautCourant[j][1][d] - xi[d]
                    dx += tmp * tmp
    
                # binning O(1)
                log_dt = np.log10(dt_ij)
                k = int((log_dt - log_min) * inv_step)
    
                if 0 <= k < nbBins:
                    MSD_par_label[labelCourant,k] += dx
                    counts[labelCourant,k] += 1
    
        MSD_par_label[labelCourant, -1] += 1
    
    
    # filtrage à la fin de la somme sur les MSD. On filtre chaque MSD par label de défaut
    msd_res= [None] * nbLabel
    dmsd_res= [None] * nbLabel
    dt_out_res= [None] * nbLabel
    
    nbdt_min=5
    for l in range(nbLabel):        
        n_ok = 0
        for k in range(nbBins):
            if counts[labelCourant,k] > nbdt_min:
                n_ok += 1
    
        msd = np.empty(n_ok)
        dmsd = np.empty(n_ok)
        dt_out = np.empty(n_ok)
            
        p = 0
        for k in range(nbBins - 1):
            if counts[labelCourant,k] > nbdt_min:
                val = MSD_par_label[labelCourant,k] / counts[labelCourant,k]
                msd[p] = val
                dmsd[p] = (val**2) / np.sqrt(counts[labelCourant,k])
                dt_out[p] = dt_centers[k]
                p += 1
                
        msd_res[l]=msd
        dmsd_res[l]=dmsd
        dt_out_res[l]=dt_out
        
    return dt_out_res, msd_res, dmsd_res


# @njit(parallel=True)
# def MSDModul3D_numba_opt(x, t):
#     Nt = x.shape[0]
#     nbBins = 100

#     # dt
#     dtmin = 0.0
#     for i in range(Nt - 1):
#         dtmin += (t[i+1] - t[i])
#     dtmin /= (Nt - 1)

#     if dtmin <= 0.0:
#         dtmin = 1e-15

#     dtmax = t[-1] * 0.5
#     if dtmax <= 0.0:
#         dtmax = 1e-15

#     log_min = np.log10(dtmin)
#     log_max = np.log10(dtmax)

#     # bins log uniformes
#     dt_bins = np.empty(nbBins)
#     step = (log_max - log_min) / (nbBins - 1)
#     for i in range(nbBins):
#         dt_bins[i] = 10.0 ** (log_min + i * step)

#     dt_centers = np.empty(nbBins - 1)
#     for i in range(nbBins - 1):
#         dt_centers[i] = np.sqrt(dt_bins[i] * dt_bins[i+1])

#     sums = np.zeros(nbBins)
#     counts = np.zeros(nbBins, dtype=np.int64)

#     max_dt = dt_bins[-1]

#     # facteur pour binning rapide
#     inv_step = 1.0 / step

#     for i in prange(Nt - 1):
#         xi = x[i]
#         ti = t[i]

#         for j in range(i + 1, Nt):
#             dt_ij = t[j] - ti
#             if dt_ij > max_dt:
#                 break

#             # calcul dx sans allocation
#             dx = 0.0
#             for d in range(x.shape[1]):   # marche pour dim > 1
#                 tmp = x[j, d] - xi[d]
#                 dx += tmp * tmp

#             # binning O(1)
#             log_dt = np.log10(dt_ij)
#             k = int((log_dt - log_min) * inv_step)

#             if 0 <= k < nbBins:
#                 sums[k] += dx
#                 counts[k] += 1

#     # filtrage
#     n_ok = 0
#     for k in range(nbBins):
#         if counts[k] > 5:
#             n_ok += 1

#     msd = np.empty(n_ok)
#     dmsd = np.empty(n_ok)
#     dt_out = np.empty(n_ok)

#     p = 0
#     for k in range(nbBins - 1):
#         if counts[k] > 5:
#             val = sums[k] / counts[k]
#             msd[p] = val
#             dmsd[p] = val / np.sqrt(counts[k])          ### Pourquoi val et pas val² dans le code d'Alizée ?
#             dt_out[p] = dt_centers[k]
#             p += 1

#     return dt_out, msd, dmsd


#Fonction calculant l'asymptote comme étant la valeur centrale du plus grand tube de diamètre val_courante*10%
def AsymptoteTube(y, eps): 
    valRes=0    #valeur de début du segment le plus long dans la plage [val-val*eps, val+val*eps]
    longRes=0   #longueur du segment le plus long    
    iRes=0
    
    for iDeb in range(len(y)):
        iP = iDeb      #pointeur courant
        compteur=0     #longueur du segment courant
        val=y[iDeb]
        
        while(iP<len(y)) and (y[iP]>val-val*eps) and (y[iP]<val+val*eps):
            compteur+=1
            if compteur>longRes:
                longRes=compteur
                valRes=y[iDeb]
                iRes=iDeb
            iP+=1
    
    return valRes, np.mean(y[iRes:iRes+longRes+1]), iRes    

def fVar(MSD, dt):
    d=aMaille                                  #distance entre les noeuds du réseau               
    Z=4                                        #nombre de plus proches voisins
    w0=Z*freqAt                                #fréquence de saut de base           ATTENTION: Faut-il bien multiplier cette fréquence par Z ?          
    D_rand = (1/Z) * d**2 * w0
    MSD_rand = 4*D_rand*dt                      
    return MSD/MSD_rand

#Calcul du facteur géométrique adimensionnée
def g():
    return (Nppv*Lppv**2)/(2*np.shape(ppv)[1])

# Fonction pour faire tourner la vue
def update(frame):
    ax.view_init(elev=20, azim=frame)
    return fig,


def truncate_colormap(cmap, minval=0.25, maxval=0.95, n=256):
    return LinearSegmentedColormap.from_list(
        f'trunc({cmap.name})',
        cmap(np.linspace(minval, maxval, n))
    )

plt.close("all")
###  Récupération et exploitation des data  ###
eps=0.05
tau = 10        #nb de valeurs sur lesquelles est moyenné D


data = np.load(f"Traj_SuiviDef_tpsVariable_N_{N}_ppv_{ppv}_cDef_{cDef}_freqDef_{freqDef}_swap_{swap}_XYZ_{X}{Y}{Z}.npz", allow_pickle=True)
pos_reelle=data["pos_reelle"]
positions=data["positions"]
temps=data["temps"]
Densite=data["Densite"]
listeDefauts=data["listeDefauts"]
labelsDefauts=data["labelsDefauts"]

with open("defauts_SuiviDef_tpsVariable_N_{N}_ppv_{ppv}_cDef_{cDef}_freqDef_{freqDef}_swap_{swap}_XYZ_{X}{Y}{Z}.pkl", "rb") as f:
    defauts = pickle.load(f)

nbDef = len(defauts.keys())
#Adimensionnement des valeurs de temps
temps*=freqAt

# #Reconstitution des trajectoires de chaque défauts, classées par fréquence de saut associée
# trajectoires=[]
# tempsTrajDef=[]
# for freq, listePos in defauts.values():
#     print("freq", freq, "listePos", listePos)
#     traj = np.empty((len(listePos), dimension))
#     tempsDef=traj = np.zeros(len(listePos))
#     for k, instant, xCourant in enumerate(listePos):
#         tempsDef[k]=instant
#         traj[k]=xCourant
#     trajectoires.append(traj)
#     tempsTrajDef.append(tempsDef)




### Calcul du MSD ###
centresMSD_labels, MSD_labels, std_labels = MSD_defauts(defauts)     

#Fitting curve_fit du MSD
std_labels[std_labels == 0] = 1e-8


nbDT = 1000

#On va traiter les MSD labels par label
for l in range(nbLabel):
    ###  On interpole la courbe du MSD obtenu en échelle log-log en une courbe similaire en échelle linéaire, à pas réguliers dans l'espace réel
    log_dt = np.linspace(np.min(np.log(centresMSD_labels[l])), np.max(np.log(centresMSD_labels[l])), num=nbDT)        
    flog_MSD = interp1d(np.log(centresMSD_labels[l]), np.log(MSD_labels[l]), kind='quadratic')
    log_MSD=flog_MSD(log_dt)
    
    new_dt=np.linspace(log_dt.min(), log_dt.max(), nbDT)     #Abscisses linéairement réparties le long de l'axe log. Idem log_dt ?     
    
    #Calcul de la dérivée, interpolation et lissage de celle-ci
    deri_log = np.gradient(log_MSD, log_dt)       
    fderi_interp_log = interp1d(log_dt, deri_log, kind='quadratic')
    deri_interp_log = fderi_interp_log(new_dt)
    
    sigma=10
    deri_lisse = gaussian_filter1d(deri_interp_log, sigma=sigma, mode='nearest')
    
    #Calcul de la pente du log du MSD
    asympTubeDeb, asympTubeMean, longTube = AsymptoteTube(deri_lisse, eps)
    
    
    ### Calcul d'un D correct avec les pentes calculées ###
    fLog_Long = np.zeros(len(centresMSD_labels[l]))
    DLog_Long = np.zeros(len(centresMSD_labels[l]))               
    
    #On redimensionne les valeurs de temps avant les calculs de f et D
    centresMSD_labels[l]/=freqAt
    
    # Coefficient de diffusion et facteur de corrélation, calculé à t pour une fenêtre de MSD centrée en t et de largeur tau
    for ind in range(len(centresMSD_labels[l])-tau//2):
        fLog_Long[ind] = np.mean(np.array([fVar(MSD_labels[l][k],centresMSD_labels[l][k]) for k in range(ind-tau//2, ind+tau//2, 1)]))
        DLog_Long[ind] = np.mean(np.array([MSD_labels[l][k]/(4*(centresMSD_labels[l][k])**asympTubeMean) for k in range(ind-tau//2, ind+tau//2, 1)]))
    
    _, D, debD = AsymptoteTube(DLog_Long, eps)
    _, f, debf = AsymptoteTube(fLog_Long, eps)
    print(f"Coefficient de diffusion obtenu pour le label {l}, cDef={cDef} et alpha={alpha}: {D} pour un tube commencé à l'indice {debD}")
    print(f"Facteur de corrélation obtenu pour le label {l}, cDef={cDef} et alpha={alpha}: {f} pour un tube commencé à l'indice {debf}")
    
    
    ###  Tracés  ###
    
    # print("(g,d,h,b)=", g,d,h,b, "   proportion de gauche/droite/haut/bas expérimentales=", gauche/(gauche+droite+haut+bas), droite/(gauche+droite+haut+bas), haut/(gauche+droite+haut+bas), bas/(gauche+droite+haut+bas))
    fig, ax = plt.subplots(2)
    # abscisses = [elem[0] for elem in positions]
    # ordonnees = [elem[1] for elem in positions]
    
    # ax[0].scatter(abscisses, ordonnees, c=range(len(abscisses)), cmap='viridis')
    # ax[0].set_title(f"Marche aléatoire 2D (g={g}, d={d}, h={h}, b={b})", fontsize=16)
    # ax[0].set_xlabel("Position horizontale", fontsize=14)
    # ax[0].set_ylabel("Position verticale", fontsize=14)
    # ax[0].tick_params(axis='both', labelsize=14)
    # fig.suptitle(f"Evolution du coefficient de diffusion et du facteur de corrélation en fonction \n du temps d'observation tau pour une trajectoire unique stockée en mémoire\n cDef={cD}, alpha={a} et N={N}", fontsize=12)
    ax[0].plot(centresMSD_labels[l], DLog_Long, '+', markersize=3, color = "red")
    ax[0].set_xlabel("dt (en secondes)", fontsize=10)
    ax[0].set_ylabel("D", fontsize=10)
    ax[0].set_title(f"D de la longue trajectoire en fonction de tau avec cDef = {cDef} et alpha = {alpha} \nswap:{swap}, tpsVar:{tpsVar} - D = {D} - concentration = {inputDef[l,1]}", fontsize=10)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")      
    ax[0].set_xlim(centresMSD_labels[l].min(),  centresMSD_labels[l].max())
    ax[0].hlines(D, centresMSD_labels[l].min(),  centresMSD_labels[l].max())
    ax[1].plot(centresMSD_labels[l], fLog_Long, '+', markersize=3, color = "blue")
    ax[1].set_xlabel("dt (en secondes)", fontsize=10)
    ax[1].set_ylabel("D", fontsize=10)
    ax[1].set_title(f"f de la longue trajectoire en fonction de tau avec cDef = {cDef} et alpha = {alpha}\n swap:{swap}, tpsVar:{tpsVar} - f = {f}", fontsize=10)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(centresMSD_labels[l].min(),  centresMSD_labels[l].max())
    ax[1].hlines(f, centresMSD_labels[l].min(),  centresMSD_labels[l].max())
    ax[0].grid("on")
    ax[1].grid("on")
    fig.subplots_adjust(hspace = 0.7) 
    fig.suptitle(f"Coefficient de diffusion D et facteur de corrélation f estimés avec cDef={cDef}, alpha={alpha}\nD={D} et f={f} \n label {l}: frequence de saut = {inputDef[l,0]}, concentration = {inputDef[l,1]}")    
    fig.savefig(f'DModul_tps_var_N_{N}_cD_{cDef}_alpha_{alpha}_sigma_{sigma}_swap_{swap}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)   # save the figure to file
    plt.close(fig)    
    
    fig, ax = plt.subplots(2)
    ax[0].plot(centresMSD_labels[l], MSD_labels[l], 'o', markersize=2, color="midnightblue", label="MSD de référence")
    ax[0].plot(np.exp(log_dt), np.exp(log_MSD), color="orchid",ls='--',lw=1, label="Courbe interpolée")
    ax[0].plot(centresMSD_labels[l], 4*D*(centresMSD_labels[l])**asympTubeMean, color="greenyellow",ls='--',lw=1, label="Courbe avec D et exposant de la loi diffusion estimés")
    ax[0].set_xlabel("dt", fontsize=9)
    ax[0].set_ylabel("MSD", fontsize=9)
    ax[0].set_title(f"MSD de référence et interpolé pour cDef={cDef} et alpha={alpha} : D={D}, f={f} et exposant de la loi={asympTubeMean} \n label {l}: frequence de saut = {inputDef[l,0]}, concentration = {inputDef[l,1]} - swap:{swap}, tpsVar:{tpsVar}", fontsize=9)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].grid("on")       
    ax[1].plot(log_dt, deri_log, '+', markersize=1, color="black", label="dérivée selon dt du MSD interpolé")
    ax[1].plot(new_dt, deri_lisse, 'o', markersize=1, color="orchid", label="dérivée selon dt lissée du MSD interpolé")
    ax[1].set_xlabel("dt", fontsize=9)
    ax[1].set_ylabel("dérivée selon dt du MSD", fontsize=9)
    ax[1].hlines(asympTubeMean, np.min(new_dt), np.max(new_dt), label=f"Valeur convergée (moyenne tube) de la dérivée du MSD à {asympTubeMean}", ls='--', color="darkcyan")            
    ax[1].set_title(f"Dérivée du MSD selon dt et valeur convergée pour cDef={cDef} et alpha={alpha} \n label {l}: frequence de saut = {inputDef[l,0]}, concentration = {inputDef[l,1]} - swap:{swap}, tpsVar:{tpsVar}", fontsize=9)
    ax[1].grid("on")
    ax[0].legend(loc = 'best', fontsize = 7) 
    ax[1].legend(loc = 'best', fontsize = 7) 
    fig.subplots_adjust(hspace = 0.5) 
    fig.savefig(f"Dérivée_MSD_Méthode_Tube_tolerance_{eps}_nbDt_{nbDT}_N_{N}_ppv_{ppv}_cD_{cDef}_alpha_{alpha}_sigma_{sigma}_swap_{swap}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)        
    plt.close(fig)
    
    
    # Marche aléatoire - Rendu 3D
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
     
    #Positions initiales des défauts
    XDisplay = [aD for (aD, bD, cD) in listeDefauts]
    YDisplay = [bD for (aD, bD, cD) in listeDefauts]
    ZDisplay = [cD for (aD, bD, cD) in listeDefauts]
    #Positions finales des défauts
    pos_defauts_Fin= np.array([valeur[1][-1][1] for valeur in defauts.values()])
    XDefCourant = [aD for (aD, bD, cD) in pos_defauts_Fin]
    YDefCourant = [bD for (aD, bD, cD) in pos_defauts_Fin]
    ZDefCourant = [cD for (aD, bD, cD) in pos_defauts_Fin]
    
    #On place les fréquences sur un axe en échelle log
    log_freqs_init = -np.log10(freqDef[labelsDefauts])
    log_freqs_final = -np.log10(freqDef[labelsDefauts])
    norm_init = plt.Normalize(vmin=np.min(log_freqs_init), vmax=np.max(log_freqs_init))
    norm_final = plt.Normalize(vmin=np.min(log_freqs_final), vmax=np.max(log_freqs_final))
    
    #Définition de colormaps chaude et froide pour les défauts initiaux et finaux respectivement
    cmap_init = truncate_colormap(plt.get_cmap('OrRd_r'))
    cmap_final = truncate_colormap(plt.get_cmap('BuGn_r'))
    
    #Récupération des positions réelles
    abscisses = [elem[0] for elem in positions.reshape(N+1, 3)]
    ordonnees = [elem[1] for elem in positions.reshape(N+1, 3)]
    hauteurs = [elem[2] for elem in positions.reshape(N+1, 3)]
    
    #Affichage des positions et défauts
    ax.scatter(abscisses, ordonnees, hauteurs, c=hauteurs, cmap='viridis', s=6, alpha=0.7)
    sc_init = ax.scatter(XDisplay,YDisplay, ZDisplay, marker="^", c=log_freqs_init, cmap=cmap_init, norm=norm_init, label="Défauts initiaux", s=6, alpha=0.5)
    sc_final = ax.scatter(XDefCourant, YDefCourant, ZDefCourant, marker="^", c=log_freqs_final, cmap=cmap_final, norm=norm_final, label="Défauts finaux", s=6, alpha=0.5)
    
    cbar_init = fig.colorbar(sc_init, ax=ax, shrink=0.5, pad=0.02)
    cbar_init.set_label("Défauts initiaux", fontsize=8)
    cbar_final = fig.colorbar(sc_final, ax=ax, shrink=0.5, pad=0.10)
    cbar_final.set_label("Défauts finaux", fontsize=8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Marche aléatoire avec N={N}, ppv={ppv} cDef = {cDef} et alpha = {alpha} - swap:{swap}, tpsVar:{tpsVar} \n label {l}: frequence de saut = {inputDef[l,0]}", fontsize=10)
    ax.legend(loc = 'best', fontsize = 8) 
    fig.savefig(f'Marche_tps_var_N_{N}_ppv_{ppv}_cD_{cDef}_alpha_{alpha}_sigma_{sigma}_swap_{swap}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)   # save the figure to file
    plt.close(fig)   
         
    
    # Densité en 3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    Densite[Densite == 0] = 1e-8        #pour éviter log(0)
    
    # Option : garder seulement les points significatifs
    threshold = np.max(Densite) * 0.02
    mask = Densite > threshold
    
    # Scatter 3D
    # Dimensions
    nx, ny, nz = Densite.shape
    # Coordonnées de la grille
    x, y, z = np.indices((nx, ny, nz))
    sc = ax.scatter(x[mask], y[mask], z[mask], c=np.log(Densite[mask]), cmap='cividis', marker='s', s=10, alpha=0.5)
    
    # Barre de couleur
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5)
    cbar.set_label('log(Densité)', fontsize=8)
    
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    sc_init = ax.scatter(XDisplay,YDisplay, ZDisplay, marker="^", c=log_freqs_init, cmap=cmap_init, norm=norm_init, label="Défauts initiaux", s=8, alpha=0.5)
    sc_final = ax.scatter(XDefCourant, YDefCourant, ZDefCourant, marker="^", c=log_freqs_final, cmap=cmap_final, norm=norm_final, label="Défauts finaux", s=8, alpha=0.5)
    
    cbar_init = fig.colorbar(sc_init, ax=ax, shrink=0.5, pad=0.02)
    cbar_init.set_label("Défauts initiaux", fontsize=8)
    cbar_final = fig.colorbar(sc_final, ax=ax, shrink=0.5, pad=0.10)
    cbar_final.set_label("Défauts finaux", fontsize=8)
    
    fig.suptitle(f"Mapping de la densité de passage de la lacune dans le réseau - N={N}, ppv={ppv} cDef = {cDef} et alpha = {alpha}\n swap:{swap}, tpsVar:{tpsVar} - label {l}: frequence de saut = {inputDef[l,0]}")
    fig.savefig(f'Densite3D_tps_var_N_{N}_ppv_{ppv}_cD_{cDef}_alpha_{alpha}_sigma_{sigma}_swap_{swap}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    
    
    ### Affichage de l'animation tournante
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle('Mapping de la densité 3D \nN = {N}, cDef = {cDef}, alpha = {alpha}, sigma = {sigma} - swap:{swap}, tpsVar:{tpsVar} \n label {l}: frequence de saut = {inputDef[l,0]} ', fontsize=10, y=0.98)
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)
    sc = ax.scatter(x[mask], y[mask], z[mask], c=np.log(Densite[mask]), cmap='viridis', marker='s', s=10, alpha=0.7)
    sc_init = ax.scatter(XDisplay,YDisplay, ZDisplay, marker="^", c=log_freqs_init, cmap=cmap_init, norm=norm_init, label="Défauts initiaux", s=8, alpha=0.5)
    sc_final = ax.scatter(XDefCourant, YDefCourant, ZDefCourant, marker="^", c=log_freqs_final, cmap=cmap_final, norm=norm_final, label="Défauts finaux", s=8, alpha=0.5)
    
    
    cbar_init = fig.colorbar(sc_init, ax=ax, shrink=0.5, pad=0.02)
    cbar_init.set_label("Défauts initiaux", fontsize=8)
    cbar_final = fig.colorbar(sc_final, ax=ax, shrink=0.5, pad=0.10)
    cbar_final.set_label("Défauts finaux", fontsize=8)
    
    
    # --- Barre de couleur pour la densité---
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5)
    cbar.set_label('log(Densité)', fontsize=8)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best', fontsize=8)
    # Animation
    ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)
    # Sauvegarde en GIF
    ani.save(f'Animation_Densite3D_tps_var_N_{N}_ppv_{ppv}_cD_{cDef}_alpha_{alpha}_sigma_{sigma}_swap_{swap}.gif', writer='pillow', fps=20, dpi=100)
    
    plt.close(fig)   
