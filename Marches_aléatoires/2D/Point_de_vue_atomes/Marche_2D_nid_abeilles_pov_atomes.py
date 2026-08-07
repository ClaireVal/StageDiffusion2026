import numpy.random as rd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp
from ase.units import kB

# Paramètres
L, H = 50, 50          # taille du réseau (indices)
N = 20000             # pas de temps
n_atomes_analyse = 100

aMaille= 1.0
G_M = 1.5
T = 300
nu_0 = 1e14
freqAt = nu_0 * exp(-G_M / (kB * T))

# Définitions des deux vecteurs de base du réseau en nid d'abeilles (honeycomb) et du vecteur permettant de passer d'un sous-réseau à l'autre (A/B)
a1 = np.array([sqrt(3)*aMaille, 0.0])
a2 = np.array([sqrt(3)*aMaille/2, 3*aMaille/2])
delta = np.array([sqrt(3)*aMaille/2, aMaille/2])

#Fonction renvoyant la position dans l'espace réelle à partir des indices en espace réduit
def site_to_cart(i, j, s):
    r = i * a1 + j * a2
    if s == 1:
        r = r + delta
    return r

#Fonction renvoyant la liste des indices des plus proches voisins de la lacune en coordonnées réduites
def voisins_reduits(i, j, s):
    if s == 0:  # A --> B
        return [(i, j, 1), (i-1, j, 1), (i, j-1, 1)]
    else:       # B --> A
        return [(i, j, 0), (i+1, j, 0), (i, j+1, 0)]

#Fonction renvoyant l'indice en coordonnées réduites
def pos_réduit(i, j):
    return i % L, j % H

# Initialisation
ix, iy, isub = 10, 10, 0   # lacune initiale

tous_sites = [(i, j, s) for i in range(L) for j in range(H) for s in (0, 1) if not (i == ix and j == iy and s == isub)] # liste de tous les sites (hors lacune)

idx = rd.choice(len(tous_sites), size=n_atomes_analyse, replace=False)      #indices choisis aléatoirement des atomes suivis
atomes_suivis = [list(tous_sites[k]) for k in idx]                          #liste des atomes suivis

pos_to_idx = {tuple(a): i for i, a in enumerate(atomes_suivis)}

# positions des atomes (dépliées) et de la lacune (réduite)
pos_atomes = np.zeros((n_atomes_analyse, N+1, 2))
pos_lacune = np.zeros((N+1, 2))

for ia, (i,j,s) in enumerate(atomes_suivis):                                #initialisation des positions des atomes dans la liste pos_atomes et de la lacune dans pos_lacune
    pos_atomes[ia,0] = site_to_cart(i,j,s)

pos_lacune[0] = site_to_cart(ix, iy, isub)


# Marche aléatoire
for k in range(1, N+1):

    voisins = voisins_reduits(ix, iy, isub)         #plus proches voisins de la lacune (il y en a 3)
    choix = rd.randint(3)
    ni, nj, ns = voisins[choix]                     #on choisit un de ces plus proches voisins au hasard

    # indices périodiques (en coordonées réduites)
    ni_red, nj_red = pos_réduit(ni, nj)

    # déplacement réel (non périodique)
    r_old = site_to_cart(ix, iy, isub)
    r_new = site_to_cart(ni, nj, ns)
    dr = r_new - r_old
    
    r_new_red = site_to_cart(ni_red, nj_red, ns)
    dr_red = r_new_red - r_old


    # par défaut, tous les atomes restent où ils sont
    pos_atomes[:, k] = pos_atomes[:, k-1]

    # échange lacune/atome suivi
    if (ni, nj, ns) in pos_to_idx:
        ia = pos_to_idx[(ni, nj, ns)]

        # déplacement inverse pour l’atome
        pos_atomes[ia, k] -= dr

        ai, aj, asub = atomes_suivis[ia]
        del pos_to_idx[(ai, aj, asub)]

        atomes_suivis[ia] = [ix, iy, isub]
        pos_to_idx[(ix, iy, isub)] = ia

    # mise à jour lacune
    pos_lacune[k] = pos_lacune[k-1] + dr_red
    ix, iy, isub = ni_red, nj_red, ns

# ── Tracé des trajectoires des atomes suivis ───────────────────────────────
plt.figure(figsize=(10, 8))

# Tracer la trajectoire de chaque atome suivi
for ia in range(n_atomes_analyse):
    x_trajectory = pos_atomes[ia, :, 0]  # Coordonnées x à chaque pas de temps
    y_trajectory = pos_atomes[ia, :, 1]  # Coordonnées y à chaque pas de temps
    plt.plot(x_trajectory, y_trajectory, marker='o', markersize=2, label=f'Atome {ia+1}')
np.savez(f"Traj_hexaBis_N_{N}.npz", pos_atomes = pos_atomes)
# Personnalisation du graphique
plt.title(f"Trajectoires des {n_atomes_analyse} atomes suivis (N={N} pas)")
plt.xlabel("Position x (cartésien)")
plt.ylabel("Position y (cartésien)")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Légende à l'extérieur pour éviter le chevauchement
plt.tight_layout()  # Ajuste la taille pour éviter les coupures

# Afficher le graphique
plt.show()
