import matplotlib.pyplot as plt
from math import exp
from ase.units import kB
import numpy as np
from numba import njit, prange
import numba
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import json


# Taille de l'espace considéré
H = 10
L = 100
x0 = [(L//2)-1,H//2]
N=100000
nbLacunes=10
nbImp = (L//2)*H

tpsVar=0       #si =0 tps constant, si =1 temps variable
swap=1          #activation du swap si 1, désactivation si 0

### Paramètres du système
# # L'ordre de grandeur pour les énergies de formation et de migration d'un défaut sont de l'ordre de 1eV (cf rapport de stage de Manon Dewynter)
G_M = 1.5 #eV = barrière d'énergie pour l'échange lacune / atome du réseau
# G_Mdef = 0.5 #eV = barrière d'énergie pour l'échange lacune / atome substitué
# G_F = 1.0 #eV = énergie libre de Gibbs de formation du défaut substitutionnel
T = 300 #K
nu_0 = 1e13

freqAt = nu_0 * exp(-G_M/(kB*T))               #Fréquence de saut des atomes du réseau
alpha = [1]


def MSDLongueVar_numba_opt_sauts(trajectoires, nbBins=100):
    """
    trajectoires : liste de tableaux 2D (nbSauts_particule, 3)
                   chaque ligne = [x, y, t] pour un saut
    """
    n_atomes = len(trajectoires)

    # Calcul de dtmin et dtmax sur toutes les trajectoires
    dtmin = 1e30
    dtmax = 0.0
    for traj in trajectoires:
        for i in range(len(traj)-1):
            dt = traj[i+1][2] - traj[i][2]
            if dt < dtmin:
                dtmin = dt
            if traj[-1][2] > dtmax:
                dtmax = traj[-1][2]
    dtmax *= 0.5

    if dtmin <= 0.0:
        dtmin = 1e-15
    if dtmax <= 0.0:
        dtmax = 1e-15

    log_min = np.log10(dtmin)
    log_max = np.log10(dtmax)
    
    print(dtmin, dtmax, log_min, log_max)

    # Bins log-uniformes
    dt_bins = np.empty(nbBins)
    step = (log_max - log_min) / (nbBins - 1)
    for i in range(nbBins):
        dt_bins[i] = 10.0 ** (log_min + i * step)

    dt_centers = np.empty(nbBins - 1)
    for i in range(nbBins - 1):
        dt_centers[i] = np.sqrt(dt_bins[i] * dt_bins[i+1])

    max_dt = dt_bins[-1]
    inv_step = 1.0 / step

    # Initialisation des accumulateurs
    sums = np.zeros(nbBins)
    counts = np.zeros(nbBins, dtype=np.int64)
    sums_rand = np.zeros(nbBins)

    # Boucle sur les atomes
    for a in range(n_atomes):
        traj = trajectoires[a]
        n_sauts = len(traj)

        sums_loc = np.zeros(nbBins)
        sums_rand_loc = np.zeros(nbBins)
        counts_loc = np.zeros(nbBins, dtype=np.int64)

        # Boucle sur les sauts de la trajectoire
        for i in range(n_sauts):
            xi = traj[i][0]
            yi = traj[i][1]
            ti = traj[i][2]

            # MSD random : somme cumulée des carrés des pas
            rand2 = 0.0

            for j in range(i+1, n_sauts):
                xj = traj[j][0]
                yj = traj[j][1]
                tj = traj[j][2]

                dt_ij = tj - ti
                if dt_ij > max_dt:
                    break

                # MSD réel
                dx2 = (xj - xi)**2 + (yj - yi)**2

                # MSD random : on ajoute le carré du pas entre j-1 et j
                if j > i+1:
                    step2 = (traj[j][0] - traj[j-1][0])**2 + (traj[j][1] - traj[j-1][1])**2
                    rand2 += step2

                # Bin
                log_dt = np.log10(dt_ij)
                k = int((log_dt - log_min) * inv_step)
                if 0 <= k < nbBins:
                    sums_loc[k] += dx2
                    sums_rand_loc[k] += rand2
                    counts_loc[k] += 1

        # Aggregation
        for k in range(nbBins):
            sums[k] += sums_loc[k]
            sums_rand[k] += sums_rand_loc[k]
            counts[k] += counts_loc[k]

    # Filtrage des bins peu peuplés
    n_ok = 0
    for k in range(nbBins):
        if counts[k] > 5:
            n_ok += 1

    msd = np.empty(n_ok)
    msd_rand = np.empty(n_ok)
    dmsd = np.empty(n_ok)
    dt_out = np.empty(n_ok)

    p = 0
    for k in range(nbBins-1):
        if counts[k] > 5:
            msd[p] = sums[k] / counts[k]
            msd_rand[p] = sums_rand[k] / counts[k]
            dmsd[p] = msd[p] / np.sqrt(counts[k])
            dt_out[p] = dt_centers[k]
            p += 1

    return dt_out, msd, dmsd, msd_rand


#Fonction calculant l'asymptote comme étant la valeur centrale du plus grand tube de diamètre val_courante*eps
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
    
    return valRes, np.mean(y[iRes:iRes+longRes+1]), longRes   

def f(MSD, tau):
    d=1                                        #distance entre les noeuds du réseau
    w0=1                                       #fréquence de saut de base             
    Z=4                                        #nombre de plus proches voisins             
    D_rand = (1/Z) * d**2 * w0     #vaut plutôt (Z/2*dim)*w0*d²
    MSD_rand = 4*D_rand*tau                      
    return MSD/MSD_rand

def fbis(MSD,tau):
    d=1                                        #distance entre les noeuds du réseau
    w0=1                                       #fréquence de saut de base             
    Z=4                                        #nombre de plus proches voisins
    MSD_rand=Z*tau*w0*d**2
    return MSD/MSD_rand

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

plt.close("all")
###  Récupération et exploitation des data  ###
eps=0.05
tau = 10        #nb de valeurs sur lesquelles est moyenné D
for a in alpha:
        data = np.load(f"Traj_lacune_2D_barre_alpha_{a}_nbLac_{nbLacunes}_N_{N}.npz", allow_pickle=True)
        positions_lacunes=data["positions_lacunes"]
        Densite=data["Densite"]
        if tpsVar==1:
            temps=data["temps"]
        else:
            temps=np.array([i for i in range(N+1)])
        
        with open(f'Traj_imp_2D_barre_alpha_{a}_nbLac_{nbLacunes}_N_{N}.json', 'r') as fc:
            traj_imp = json.load(fc)
        
        print("tttttt", [pos[0] for pos in positions_lacunes])
        



        ### Calcul du MSD ###
        centresMSD, MSDLogVar, stdLogVar, MSDRand = MSDLongueVar_numba_opt_sauts(traj_imp)     
        
        #Fitting curve_fit du MSD
        stdLogVar[stdLogVar == 0] = 1e-8
        
        
        nbDT = 1000
        
        ###  On interpole la courbe du MSD obtenu en échelle log-log en une courbe similaire en échelle linéaire, à pas réguliers dans l'espace réel
        log_dt = np.linspace(np.min(np.log(centresMSD)), np.max(np.log(centresMSD)), num=nbDT)        
        flog_MSD = interp1d(np.log(centresMSD), np.log(MSDLogVar), kind='quadratic')
        log_MSD=flog_MSD(log_dt)
        
        new_dt=np.linspace(log_dt.min(), log_dt.max(), nbDT)     #Abscisses linéairement réparties le long de l'axe log. Idem log_dt ?     
        
        #Calcul de la dérivée, interpolation et lissage de celle-ci
        deri_log = np.gradient(log_MSD, log_dt)       
        fderi_interp_log = interp1d(log_dt, deri_log, kind='quadratic')
        deri_interp_log = fderi_interp_log(new_dt)
        
        sigma=20
        deri_lisse = gaussian_filter1d(deri_interp_log, sigma=sigma, mode='nearest')
        
        #Calcul de la pente du log du MSD
        asympTubeDeb, asympTubeMean, longTube = AsymptoteTube(deri_lisse, eps)
        
        
        fLog_Long = np.zeros(len(centresMSD))
        DLog_Long = np.zeros(len(centresMSD))

        for ind in range(len(centresMSD) - tau//2):
            DLog_Long[ind] = np.mean(np.array([MSDLogVar[k] / (4 * (centresMSD[k])**asympTubeMean) for k in range(ind - tau//2, ind + tau//2, 1)]))
            
            lo = max(0, ind - tau//2)
            hi = ind + tau//2
            num = MSDLogVar[lo:hi]
            den = MSDRand[lo:hi]
            
            mask = np.isfinite(num) & np.isfinite(den) & (den > 0)
            
            if np.any(mask):
                fLog_Long[ind] = np.sum(num[mask]) / np.sum(den[mask])
            else:
                fLog_Long[ind] = np.nan
            
         
        _, D, lenD = AsymptoteTube(DLog_Long, eps)
        _, fCor, lenf = AsymptoteTube(fLog_Long, eps)
        print(f"Coefficient de diffusion obtenu pour alpha={a}: {D}")
        print(f"Facteur de corrélation obtenu pour alpha={a}: {fCor}")
        
        # ## Calcul des incertitudes et comparaison avec un fit linéaire classique
        
        # #Incertitudes fit linéaire
        # aL, sigmaL, bL, sigmabL = fit_lineaire(np.log(centresMSD), np.log(MSDLogVar))
        # Dfit = np.exp(bL)/4
        # incert_D = 0.5 * np.exp(bL) * sigmabL
        # print("D via fit lineaire =", Dfit)
        # print("l'incertitude sur D pour le fit linéaire simple est de :", incert_D)
    
        # #Incertitudes méthode du tube
        # incert_D_tube = 0.05*(D/np.sqrt(lenD))
        # print("D via méth. tube =", D)
        # print("l'incertitude sur D pour la méthode tube est de:", incert_D_tube)
        
        
        
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
        ax[0].plot(centresMSD, DLog_Long, '+', markersize=3, color = "red")
        ax[0].set_xlabel("dt (en secondes)", fontsize=11)
        ax[0].set_ylabel("D", fontsize=11)
        ax[0].set_title(f"D de la longue trajectoire en fonction de tau avec alpha = {a} \nTemps de saut variable - D = {D}", fontsize=10)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")      
        ax[0].hlines(D, centresMSD.min(),  centresMSD.max(), color="firebrick")
        ax[1].plot(centresMSD, fLog_Long, '+', markersize=3, color = "blue")
        ax[1].hlines(fCor, centresMSD.min(),  centresMSD.max(), color="navy")
        ax[1].set_xlabel("dt (en secondes)", fontsize=11)
        ax[1].set_ylabel("D", fontsize=11)
        ax[1].set_title(f"f de la longue trajectoire en fonction de tau avec alpha = {a}\nFréquence de saut variable", fontsize=10)
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        ax[0].grid("on")
        ax[1].grid("on")
        fig.subplots_adjust(hspace = 0.7)     
        fig.savefig(f'D_f_barre_swap_{swap}_tpsVar_{tpsVar}_N_{N}_alpha_{a}_sigma_{sigma}_eps_{eps}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)   # save the figure to file
        plt.close(fig)    
        
        fig, ax = plt.subplots(2)
        ax[0].plot(centresMSD, MSDLogVar, 'o', markersize=2, color="midnightblue", label="MSD de référence")
        ax[0].plot(np.exp(log_dt), np.exp(log_MSD), color="orchid",ls='--',lw=1, label="Courbe interpolée")
        ax[0].plot(centresMSD, 4*D*(centresMSD)**asympTubeMean, color="greenyellow",ls='--',lw=1, label="Courbe avec D et exposant de la loi diffusion estimés")
        ax[0].set_xlabel("dt", fontsize=9)
        ax[0].set_ylabel("MSD", fontsize=9)
        ax[0].set_title(f"MSD de référence et interpolé pour alpha={a}\n sigma de l'interpolation = {sigma}, D={D} et exposant de la loi={asympTubeMean}", fontsize=9)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].grid("on")       
        ax[1].plot(log_dt, deri_log, '+', markersize=1, color="black", label="dérivée selon dt du MSD interpolé")
        ax[1].plot(new_dt, deri_lisse, 'o', markersize=1, color="orchid", label="dérivée selon dt lissée du MSD interpolé")
        ax[1].set_xlabel("dt", fontsize=9)
        ax[1].set_ylabel("dérivée selon dt du MSD", fontsize=9)
        ax[1].hlines(asympTubeMean, np.min(new_dt), np.max(new_dt), label=f"Valeur convergée (moyenne tube) de la dérivée du MSD à {asympTubeMean}", ls='--', color="darkcyan")            
        ax[1].set_title(f"Dérivée du MSD selon dt et valeur convergée calculée par la méthode du tube pour un nombre d'abscisses\n d'interpolation du MSD nb_dt={nbDT}, sigma={sigma} et alpha={a} - Fréquence de saut variable", fontsize=9)
        ax[1].grid("on")
        ax[0].legend(loc = 'best', fontsize = 6) 
        ax[1].legend(loc = 'best', fontsize = 6) 
        fig.subplots_adjust(hspace = 0.5) 
        fig.savefig(f"Dérivée_barre_MSD_Méthode_Tube_tolerance_{eps}_nbDt_{nbDT}_N_{N}_alpha_{a}_sigma_{sigma}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)        
        plt.close(fig)
        
        # fig, ax = plt.subplots()
        # abscisses = [elem[0] for elem in positions]
        # ordonnees = [elem[1] for elem in positions]
        # ax.scatter(abscisses, ordonnees, c=range(len(abscisses)), cmap='viridis')
        # ax.set_xlabel("positions X", fontsize=11)
        # ax.set_ylabel("positions Y", fontsize=11)
        # ax.set_title(f"Marche aléatoire avec alpha = {a} \nFréquence de saut variable", fontsize=11)
        # fig.savefig(f'Marche_swap_tps_var_N_{N}_alpha_{a}_sigma_{sigma}_eps_{eps}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)   # save the figure to file
        # plt.close(fig)   
        
        #Marches des impuretés
        fig, ax = plt.subplots()
        for ia in range(len(traj_imp)):
            x_trajectory = [pos[0] for pos in traj_imp[ia]]  # Coordonnées x à chaque pas de temps
            y_trajectory = [pos[1] for pos in traj_imp[ia]]  # Coordonnées y à chaque pas de temps
            plt.plot(x_trajectory, y_trajectory, marker='o', markersize=2, label=f'Impureté {ia+1}')
        ax.set_xlabel("positions X", fontsize=11)
        ax.set_ylabel("positions Y", fontsize=11)
        ax.set_title(f"Marche aléatoire avec alpha = {a} \nFréquence de saut variable", fontsize=11)
        fig.savefig(f'Marche_barre_impuretés_N_{N}_alpha_{a}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)   # save the figure to file
        plt.close(fig)  
        
        
        
        Densite[Densite == 0] = 1e-8        #pour éviter log(0)
        fig, ax = plt.subplots()
        im = ax.imshow(np.log(Densite.T), cmap='plasma', origin='lower')
        fig.colorbar(im, ax=ax, label="log(Densité de passage)")
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        
        ax.set_title(f"Mapping de la densité de passage pour alpha ={a}, \n et position initiale {x0}\nFréquence de saut variable", fontsize=9)
        ax.tick_params(axis='both', labelsize=11)
        
        # XDisplay = []
        # YDisplay = []
        # for (aDef,bDef) in listeDefauts:
        #     XDisplay.append(aDef)
        #     YDisplay.append(bDef)
        # plt.plot(XDisplay, YDisplay, "x", markersize=2, color="dodgerblue", label='Positions initiales défauts',alpha=0.6) 
            
        # if swap==1:  
        #     XDefCourant = []
        #     YDefCourant = []
        #     for (aD,bD) in defautsFin:
        #         XDefCourant.append(aD)
        #         YDefCourant.append(bD)
        #     plt.plot(XDefCourant, YDefCourant, "^", markersize=2, color="firebrick", label='Positions finales défauts', alpha=0.6)
        
        # plt.plot(abscisses, ordonnees,color='k',marker='o',ls='none',alpha=0.1)
        
        # plt.legend(prop={'size':8}, loc='lower left')
        fig.savefig(f'Densite_barre_N_{N}__alpha_{a}_.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)   
        

        
