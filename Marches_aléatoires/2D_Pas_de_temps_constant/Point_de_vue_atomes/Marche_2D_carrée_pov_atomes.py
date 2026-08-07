import numpy.random as rd
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import exp, floor
from ase.units import kB
import numpy as np
from matplotlib.animation import FuncAnimation

mpl.rcParams['figure.figsize']  = (12, 12)

plt.close("all")
# Taille de l'espace considéré
H = 100
L = 100
Natomes = H*L
N=10000             #nb de pas
swap=1              #activation du swap si 1, désactivation si 0
tpsVar=0            #si =0 tps constant, si =1 temps variable

G_M = 1.5  #eV
T= 300   #K
nu_0 = 1e13  #Hz

freqAt = nu_0 * exp(-G_M / (kB * T))
cDef = [0.0001, 0.001,0.01, 0.1,0.5]
alpha = [0.001,0.01,0.1,1,10, 100, 1000]

x0 = [33, 30]   # coordonnées cartésiennes réelles de la lacune
aMaille = 1  #Angstrom

#Nb d'atomes du réseau dont on va stocker et étudier la trajectoire pour le calcul de D et f en auto-diffusion
n_atomes_analyse = 100


# ── Conversion indices → cartésien ───────────────────────────────────────────
def site_to_cart(i, j):
    return np.array([i * aMaille, j * aMaille], dtype='float64')


# ── Conversion cartésien → indices (pour retrouver le site de x0) ────────────
def cart_to_site(x_cart, y_cart):
    i = int(round(x_cart / aMaille))
    j = int(round(y_cart / aMaille))
    return i, j


# Fonction renvoyant les 3 voisins en coordonnées réduites (AVANT modulo)
def voisins_reduits(i, j):
    return [
        (i+1, j),  # droite
        (i-1, j),  # gauche
        (i, j+1),  # haut
        (i, j-1)   # bas
    ]

# Fonction permettant de connaître l'environnement, s'il y a des défauts dans cet environnement, et si oui où
def environnement():
    env = np.array([
        ( aMaille,  0.0),  # droite
        (-aMaille,  0.0),  # gauche
        ( 0.0,  aMaille),  # haut
        ( 0.0, -aMaille)   # bas
    ], dtype='float64')
    testDef    = False
    voisinsDef = []
    return env, testDef, voisinsDef

# ═════════════════════════════════════════════════════════════════════════════

compteur = 0

for a in alpha:
    for cD in cDef:

        freqDef = a*freqAt
        Ndef = floor(cD*Natomes)

        listeDefauts = []
        for _ in range(Ndef):
            listeDefauts.append((int(rd.random()*L), int(rd.random()*H)))
        defauts = set(listeDefauts)

        #  Position initiale de la lacune en indices réduits
        i0, j0 = cart_to_site(x0[0], x0[1])
        ix, iy = i0, j0   # coordonnées réduites courantes de la lacune

        # Initialisation des trajectoires stockées et densité
        Densite = np.zeros((L, H))
        positions = np.zeros((N+1, 2))    # cartésien réduit
        pos_reelle = np.zeros((N+1, 2))    # cartésien déplié

        pos_reelle[0] = x0
        positions[0]  = site_to_cart(ix, iy)

        # Sélection des atomes suivis
        tous_sites = [(i, j) for i in range(L) for j in range(H) if (i, j) != (i0, j0)]
        idx_choisis = rd.choice(len(tous_sites), size=n_atomes_analyse, replace=False)

        # Coordonnées réduites courantes de chaque atome suivi
        atomes_suivis = [list(tous_sites[k]) for k in idx_choisis]

        # Trajectoires cartésiennes dépliées : initialisées avec site_to_cart
        pos_atomes = np.zeros((n_atomes_analyse, N+1, 2))
        for ia, (si, sj) in enumerate(atomes_suivis):
            pos_atomes[ia, 0] = site_to_cart(si, sj)  # Position initiale
            
        
        # Lookup position réduite → indice atome
        pos_to_idx = {tuple(at): ia for ia, at in enumerate(atomes_suivis)}

        # ── Marche aléatoire ──────────────────────────────────────────────
        for k in range(1, N+1):
            envPPV = environnement()[0]

            gamma = np.ones(4) * freqAt
            gamma /= np.sum(gamma)
            
            u = rd.random()
            cum = np.cumsum(gamma)
            
            num_saut = np.searchsorted(cum, u)
            

            dx, dy = envPPV[num_saut]
            ni, nj = voisins_reduits(ix, iy)[num_saut]
            vois_red = (ni % L, nj % H)          # conditions aux limites

            # Échange lacune / atome
            # Par défaut : tous les atomes restent où ils sont
            for ia in range(n_atomes_analyse):
                pos_atomes[ia, k] = pos_atomes[ia, k-1]
            
            # Si échange lacune / atome suivi
            if vois_red in pos_to_idx:
                compteur +=1
                ind_atome = pos_to_idx[vois_red]
            
                pos_atomes[ind_atome, k] -= np.array([dx, dy])
            
                ai, aj = atomes_suivis[ind_atome]
                del pos_to_idx[(ai, aj)]
                atomes_suivis[ind_atome] = [ix % L, iy % H]
                pos_to_idx[(ix % L, iy % H)] = ind_atome

            # Mise à jour de la lacune
            pos_reelle[k] = pos_reelle[k-1] + np.array([dx, dy])
            ix, iy = vois_red
            positions[k]  = site_to_cart(ix, iy)
            Densite[ix, iy] += 1
        
        # ── Sauvegarde ────────────────────────────────────────────────────
        np.savez(f"Traj_carrée_swap_{swap}_tpsVariable_{tpsVar}_N_{N}_cD_{cD}_alpha_{a}.npz", pos_reelle = pos_reelle, Densite = Densite, positions = positions, listeDefauts = listeDefauts, pos_atomes = pos_atomes)
        print("compteur", compteur)
        print("compteur/N", compteur/N)
        
        # ── Tracé des trajectoires des atomes suivis ───────────────────────────────
        plt.figure(figsize=(10, 8))
        
        # Tracer la trajectoire de chaque atome suivi
        for ia in range(n_atomes_analyse):
            x_trajectory = pos_atomes[ia, :, 0]  # Coordonnées x à chaque pas de temps
            y_trajectory = pos_atomes[ia, :, 1]  # Coordonnées y à chaque pas de temps
            plt.plot(x_trajectory, y_trajectory, marker='o', markersize=2, label=f'Atome {ia+1}')
        
        # Personnalisation du graphique
        plt.title(f"Trajectoires des {n_atomes_analyse} atomes suivis (N={N} pas)")
        plt.xlabel("Position x (cartésien)")
        plt.ylabel("Position y (cartésien)")
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Légende à l'extérieur pour éviter le chevauchement
        plt.tight_layout()  # Ajuste la taille pour éviter les coupures
        plt.savefig(f"Marche_carrée_suivi_atome_cD_{cD}_alpha_{a}_N_{N}.png")
        
        
       # ################ ANIMATION ################################################################### 
        


       #  nbAtomes, nbPas, _ = pos_atomes.shape

       #  fig, ax = plt.subplots(figsize=(8, 8))
        
       #  xmin = min(pos_atomes[:, :, 0].min(), positions[:, 0].min())
       #  xmax = max(pos_atomes[:, :, 0].max(), positions[:, 0].max())
       #  ymin = min(pos_atomes[:, :, 1].min(), positions[:, 1].min())
       #  ymax = max(pos_atomes[:, :, 1].max(), positions[:, 1].max())
        
       #  ax.set_xlim(xmin, xmax)
       #  ax.set_ylim(ymin, ymax)
       #  ax.set_xlabel("x")
       #  ax.set_ylabel("y")
       #  ax.set_title("Vacancy-mediated diffusion in 2D cubic lattice")
       #  ax.grid(True)
        
       #  # --- scatter des atomes ---
       #  scat_atomes = ax.scatter(
       #      pos_atomes[:, 0, 0],
       #      pos_atomes[:, 0, 1],
       #      s=20,
       #      c='black',
       #      label='Tracer atom'
       #  )
        
       #  # --- scatter de la lacune / défaut ---
       #  scat_defaut = ax.scatter(
       #      positions[0, 0],
       #      positions[0, 1],
       #      s=80,
       #      c='blue',
       #      marker='x',
       #      label='Vacancy'
       #  )
        
       #  ax.legend(fontsize = 10)
        
       #  def update(frame):
       #      scat_atomes.set_offsets(pos_atomes[:, frame, :])
       #      scat_defaut.set_offsets(positions[frame])
       #      ax.set_title(f"Time step : {frame}")
       #      return scat_atomes, scat_defaut
        
       #  # Slow down the animation by increasing the interval and decreasing the fps
       #  ani = FuncAnimation(
       #      fig,
       #      update,
       #      frames=range(0, nbPas, 20),  # Adjust the step size if needed
       #      interval=50,  # Increased from 10ms to 200ms (slower)
       #      blit=False
       #  )
        
       #  ani.save(
       #      "diffusion_lacune.gif",
       #      writer="pillow",
       #      fps=5,  # Decreased from 10 fps to 5 fps (slower)
       #      dpi=100
       #  )
        
       #  #plt.show()
