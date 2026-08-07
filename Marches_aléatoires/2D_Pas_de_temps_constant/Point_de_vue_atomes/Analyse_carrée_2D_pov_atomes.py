import matplotlib.pyplot as plt
from math import exp
from ase.units import kB
import numpy as np
from numba import njit, prange
import numba
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


H = 100
L = 100
x0 = [50,50]
N = 10000

tpsVar = 0
swap = 1

### Paramètres du système
G_M = 1.5       #eV
T = 300         #K
nu_0 = 1e13

freqAt = nu_0 * exp(-G_M/(kB*T))
cDef = [0.0001, 0.001,0.01, 0.1,0.5]
alpha = [0.001,0.01,0.1,1,10, 100, 1000]

n_atomes_analyse = 100 # Nombre d'atomes sur lesquels moyenner le MSD

print("Threads actifs :", numba.get_num_threads())


@njit
def MSDLongueVar_numba_opt_atomes(x_atomes, t):
    
    n_atomes = x_atomes.shape[0]
    Nt = x_atomes.shape[1]
    nbBins = 100

    # Calcul de dtmin comme moyenne des pas de temps
    dtmin = 0.0
    for i in range(Nt - 1):
        dtmin += (t[i+1] - t[i])
    dtmin /= (Nt - 1)
    if dtmin <= 0.0:
        dtmin = 1e-15

    dtmax = t[-1] * 0.5
    if dtmax <= 0.0:
        dtmax = 1e-15

    log_min = np.log10(dtmin)
    log_max = np.log10(dtmax)

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


    # Boucle sur les atomes — parallélisation sur les atomes plutôt que
    # sur i pour éviter les race conditions sur sums/counts
    
    # MSD réel
    sums = np.zeros(nbBins)
    counts = np.zeros(nbBins, dtype=np.int64)
    
    # MSD random
    sums_rand = np.zeros(nbBins)
    
    for a in prange(n_atomes):
    
        x = x_atomes[a]
    
        sums_loc = np.zeros(nbBins)
        sums_rand_loc = np.zeros(nbBins)
        counts_loc = np.zeros(nbBins, dtype=np.int64)
    
        for i in range(Nt-1):
    
            xi = x[i]
            ti = t[i]
    
            # Somme cumulée des carrés des sauts
            rand2 = 0.0
    
            for j in range(i+1, Nt):
    
                dt_ij = t[j] - ti
    
                if dt_ij > max_dt:
                    break
    
                # -------- MSD réel --------
    
                dx2 = 0.0
                for d in range(2):
                    tmp = x[j,d] - xi[d]
                    dx2 += tmp*tmp
    
                # -------- MSD random --------
    
                step2 = 0.0
                for d in range(2):
                    step = x[j,d] - x[j-1,d]
                    step2 += step*step
    
                rand2 += step2
    
                # -------- Bin --------
    
                log_dt = np.log10(dt_ij)
                k = int((log_dt-log_min)*inv_step)
    
                if 0 <= k < nbBins:
                    sums_loc[k] += dx2
                    sums_rand_loc[k] += rand2
                    counts_loc[k] += 1
    
        for k in range(nbBins):
            sums[k] += sums_loc[k]
            sums_rand[k] += sums_rand_loc[k]
            counts[k] += counts_loc[k]

    print("Threads actifs dans fonction :", numba.get_num_threads())

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
    
            msd[p] = sums[k]/counts[k]
            msd_rand[p] = sums_rand[k]/counts[k]
    
            dmsd[p] = msd[p]/np.sqrt(counts[k])
    
            dt_out[p] = dt_centers[k]
    
            p += 1
    
    return dt_out, msd, dmsd, msd_rand


def AsymptoteTube(y, eps):
    valRes  = 0
    longRes = 0
    iRes = 0

    for iDeb in range(len(y)):
        iP = iDeb
        compteur = 0
        val = y[iDeb]

        while (iP < len(y)) and (y[iP] > val - val*eps) and (y[iP] < val + val*eps):
            compteur += 1
            if compteur > longRes:
                longRes = compteur
                valRes = y[iDeb]
                iRes = iDeb
            iP += 1

    return valRes, np.mean(y[iRes:iRes+longRes+1]), iRes


# def f(MSD, tau):
#     d = 1
#     w0 = 1/(L*H -1)             #différente de 1, car contrairement au cas de la lacune, il n'y a pas un déplacement par pas de temps, mais beaucoup  (vaut la probabilité qu'un pas de temps soit associé au déplacement de l'un des atomes suivis)
#     # Z = 3
#     # D_rand = (1/Z) * d**2 * w0
#     # D_rand = d**2 * w0
#     # MSD_rand = 4 * D_rand * tau      #retirer le 4 ? cf. p63 book diffusion   MSD=(1/(L*H -1))*d² flop :/
#     MSD_rand = w0*tau*(d**2)           #correspond à la formule p.61 du bookDiffusion : MSD_rand = <n>d², avec <n>d²=w0*tau ici
#     return MSD / MSD_rand

# def f(MSD, tau, MSDRand):
#     return MSD / MSDRand

def lineaire(t, x, y):
    return x + t*y


plt.close("all")

eps = 0.05
tau = 10

for a in alpha:
    for cD in cDef:
        data = np.load(f"Traj_carrée_swap_{swap}_tpsVariable_{tpsVar}_N_{N}_cD_{cD}_alpha_{a}.npz", allow_pickle=True)
        pos_atomes = data["pos_atomes"]   # (n_atomes_total, N_pas, 2), coordonnées dépliées
        # Densite = data["Densite"]
        
        if tpsVar == 1:
            temps = data["temps"]
        else:
            temps = np.array([i for i in range(N+1)])

        # Sélection aléatoire de n_atomes_analyse atomes parmi tous
        n_total = pos_atomes.shape[0]
        indices = np.random.choice(n_total, size=min(n_atomes_analyse, n_total), replace=False)
        nbSuivis = len(indices)
        print("Nombre d'atomes du réseau hexagonal suivis :", nbSuivis)
        pos_select = pos_atomes[indices]    # (n_atomes_analyse, N_pas, 2)

        ### Calcul du MSD moyenné sur les atomes sélectionnés ###
        centresMSD, MSDLogVar, stdLogVar, msdRand = MSDLongueVar_numba_opt_atomes(pos_select, temps)

        stdLogVar[stdLogVar == 0] = 1e-8

        nbDT = 1000

        log_dt = np.linspace(np.min(np.log(centresMSD)), np.max(np.log(centresMSD)), num=nbDT)
        flog_MSD = interp1d(np.log(centresMSD), np.log(MSDLogVar), kind='quadratic')
        log_MSD = flog_MSD(log_dt)

        new_dt = np.linspace(log_dt.min(), log_dt.max(), nbDT)

        deri_log = np.gradient(log_MSD, log_dt)
        fderi_interp_log = interp1d(log_dt, deri_log, kind='quadratic')
        deri_interp_log = fderi_interp_log(new_dt)

        sigma = 20
        deri_lisse = gaussian_filter1d(deri_interp_log, sigma=sigma, mode='nearest')

        asympTubeDeb, asympTubeMean, longTube = AsymptoteTube(deri_lisse, eps)

        fLog_Long = np.zeros(len(centresMSD))
        DLog_Long = np.zeros(len(centresMSD))

        for ind in range(len(centresMSD) - tau//2):
            DLog_Long[ind] = np.mean(np.array([MSDLogVar[k] / (4 * (centresMSD[k])**asympTubeMean) for k in range(ind - tau//2, ind + tau//2, 1)]))
            
            lo = max(0, ind - tau//2)
            hi = ind + tau//2
            num = MSDLogVar[lo:hi]
            den = msdRand[lo:hi]
            
            mask = np.isfinite(num) & np.isfinite(den) & (den > 0)
            
            if np.any(mask):
                fLog_Long[ind] = np.sum(num[mask]) / np.sum(den[mask])
            else:
                fLog_Long[ind] = np.nan
            
        _, D, _ = AsymptoteTube(DLog_Long, eps)
        _, f, _ = AsymptoteTube(fLog_Long, eps)
        
        

        print(f"Coefficient de diffusion obtenu pour cDef={cD} et alpha={a}: {D}")
        print(f"Facteur de corrélation obtenu pour cDef={cD} et alpha={a}: {f}")
        
        plt.loglog(centresMSD, MSDLogVar, label="MSD")
        plt.loglog(centresMSD, msdRand, label="MSD_rand", linestyle="--")
        plt.xlabel("Temps (log)")
        plt.ylabel("MSD (log)")
        plt.title(f"Structure nid d'abeilles 2D - f={f}")
        plt.legend()
        plt.show()
        
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
        ax[0].set_title(f"D de la longue trajectoire en fonction de tau avec cDef = {cD} et alpha = {a} \nTemps de saut variable - D = {D}", fontsize=10)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")      
        ax[0].hlines(D, centresMSD.min(),  centresMSD.max(), color="firebrick")
        ax[1].plot(centresMSD, fLog_Long, '+', markersize=3, color = "blue", label="Facteur de corrélation calculé via les D")
        ax[1].hlines(f, centresMSD.min(),  centresMSD.max(), color="navy")
        ax[1].set_xlabel("dt (en secondes)", fontsize=11)
        ax[1].set_ylabel("D", fontsize=11)
        ax[1].set_title(f"f de la longue trajectoire en fonction de tau avec cDef = {cD} et alpha = {a}\nFréquence de saut variable", fontsize=10)
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        ax[0].grid("on")
        ax[1].grid("on")
        ax[1].legend(loc = 'best', fontsize = 6) 
        fig.subplots_adjust(hspace = 0.7)     
        fig.savefig(f'D_f_carrée_swap_{swap}_tpsVar_{tpsVar}_N_{N}_cD_{cD}_alpha_{a}_sigma_{sigma}_eps_{eps}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)   # save the figure to file
        plt.close(fig)    
        
        fig, ax = plt.subplots(2)
        ax[0].plot(centresMSD, MSDLogVar, 'o', markersize=2, color="midnightblue", label="MSD de référence")
        ax[0].plot(np.exp(log_dt), np.exp(log_MSD), color="orchid",ls='--',lw=1, label="Courbe interpolée")
        ax[0].plot(centresMSD, 4*D*(centresMSD)**asympTubeMean, color="greenyellow",ls='--',lw=1, label="Courbe avec D et exposant de la loi diffusion estimés")
        ax[0].set_xlabel("dt", fontsize=9)
        ax[0].set_ylabel("MSD", fontsize=9)
        ax[0].set_title(f"MSD de référence et interpolé pour cDef={cD} et alpha={a}\n sigma de l'interpolation = {sigma}, D={D} et exposant de la loi={asympTubeMean}", fontsize=9)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].grid("on")       
        ax[1].plot(log_dt, deri_log, '+', markersize=1, color="black", label="dérivée selon dt du MSD interpolé")
        ax[1].plot(new_dt, deri_lisse, 'o', markersize=1, color="orchid", label="dérivée selon dt lissée du MSD interpolé")
        ax[1].set_xlabel("dt", fontsize=9)
        ax[1].set_ylabel("dérivée selon dt du MSD", fontsize=9)
        ax[1].hlines(asympTubeMean, np.min(new_dt), np.max(new_dt), label=f"Valeur convergée (moyenne tube) de la dérivée du MSD à {asympTubeMean}", ls='--', color="darkcyan")            
        ax[1].set_title(f"Dérivée du MSD selon dt et valeur convergée calculée par la méthode du tube pour un nombre d'abscisses\n d'interpolation du MSD nb_dt={nbDT}, sigma={sigma}, cDef={cD} et alpha={a} - Fréquence de saut variable", fontsize=9)
        ax[1].grid("on")
        ax[0].legend(loc = 'best', fontsize = 6) 
        ax[1].legend(loc = 'best', fontsize = 6) 
        fig.subplots_adjust(hspace = 0.5) 
        fig.savefig(f"Dérivée_carrée_swap_MSD_Méthode_Tube_tolerance_{eps}_nbDt_{nbDT}_N_{N}_cD_{cD}_alpha_{a}_sigma_{sigma}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)        
        plt.close(fig)
        
        fig, ax = plt.subplots()
        for ia in range(nbSuivis):
            x_trajectory = pos_atomes[ia, :, 0]  # Coordonnées x à chaque pas de temps
            y_trajectory = pos_atomes[ia, :, 1]  # Coordonnées y à chaque pas de temps
            plt.plot(x_trajectory, y_trajectory, marker='o', markersize=2, label=f'Atome {ia+1}')
        ax.set_xlabel("positions X", fontsize=11)
        ax.set_ylabel("positions Y", fontsize=11)
        ax.set_title(f"Marche aléatoire avec cDef = {cD} et alpha = {a} \nFréquence de saut variable", fontsize=11)
        fig.savefig(f'Marche_carrée_swap_tps_var_N_{N}_cD_{cD}_alpha_{a}_sigma_{sigma}_eps_{eps}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)   # save the figure to file
        plt.close(fig)   
        
        # Densite[Densite == 0] = 1e-8        #pour éviter log(0)
        # fig, ax = plt.subplots()
        # im = ax.imshow(np.log(Densite.T), cmap='plasma', origin='lower')
        # fig.colorbar(im, ax=ax, label="log(Densité de passage)")
        # ax.set_xlabel('X', fontsize=9)
        # ax.set_ylabel('Y', fontsize=9)
        
        # ax.set_title(f"Mapping de la densité de passage pour alpha ={a}, concentration de défauts = {cD}\n et position initiale {x0}\nFréquence de saut variable", fontsize=9)
        # ax.tick_params(axis='both', labelsize=11)
        
        # XDisplay = []
        # YDisplay = []
        # plt.plot(XDisplay, YDisplay, "x", markersize=2, color="dodgerblue", label='Positions initiales défauts',alpha=0.6) 
            
        # # if swap==1:  
        # #     XDefCourant = []
        # #     YDefCourant = []
        # #     for (aD,bD) in defautsFin:
        # #         XDefCourant.append(aD)
        # #         YDefCourant.append(bD)
        # #     plt.plot(XDefCourant, YDefCourant, "^", markersize=2, color="firebrick", label='Positions finales défauts', alpha=0.6)
        
        # # plt.plot(abscisses, ordonnees,color='k',marker='o',ls='none',alpha=0.1)
        
        # plt.legend(prop={'size':8}, loc='lower left')
        # fig.savefig(f'Densite_hexa_swap_tps_var_N_{N}_cD_{cD}_alpha_{a}_sigma_{sigma}_eps_{eps}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        # plt.close(fig)  
