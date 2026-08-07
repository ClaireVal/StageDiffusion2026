import numpy.random as rd
# import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import exp, log
from ase.units import kB
import numpy as np
import json
import matplotlib.animation as animation
from scipy.stats import gaussian_kde



#mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#mpl.rc('text', usetex=True)
mpl.rcParams['ytick.labelsize']=14
mpl.rcParams['xtick.labelsize']=14
#mpl.rcParams['text.fontsize']=20

mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['figure.figsize'] = (12, 12)

# Pour éviter une erreur de "exceeded cell block limit in Agg"
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['path.simplify_threshold'] = 1.0

plt.close("all")
# Taille de l'espace considéré
H = 10
L = 100
N=1000000            #nb de pas
nbImp = (L//2)*H     #nb d'impuretés
nbLacunes = 10       # Nombre de lacunes
swap=1             #activation du swap si 1, désactivation si 0
tpsVar=1           #si =1 temps variable (et si =0 tps constant, mais code uniquement pour temps variable)

#Paramètres pour les animations
pas_anim = N//20        #Pas de temps de l'animation
nbBarresHisto = 50      #Nombre de barres de l'histogramme
borneMaxHisto = (L*H)//50 +2     #Borne max de l'axe vertical de l'histogramme
temps_frame = []        #Pas auxquels les frames sont enregistrées

### Paramètres du système
# # L'ordre de grandeur pour les énergies de formation et de migration d'un défaut sont de l'ordre de 1eV (cf rapport de stage de Manon Dewynter)
G_M = 1.5 #eV = barrière d'énergie pour l'échange lacune / atome du réseau
# G_Mdef = 0.5 #eV = barrière d'énergie pour l'échange lacune / atome substitué
# G_F = 1.0 #eV = énergie libre de Gibbs de formation du défaut substitutionnel
G_Mlac = 1.5 #eV = barrière d'énergie pour l'interaction / l'échange entre deux lacunes voisines
T = 300 #K
nu_0 = 1e13
freqAt = nu_0 * exp(-G_M/(kB*T))               #Fréquence de saut des atomes du réseau
freqLacLac = nu_0 * exp(-G_Mlac/(kB*T))        #Fréquence de saut modifiée quand une lacune voisine est présente (interaction lacune-lacune)

alpha = [1000, 100, 10, 1, 0.1, 0.01, 0.001]
beta = [100, 1000, 0.001]


# Positions initiales des lacunes 
x0 = np.zeros((nbLacunes,2))
for i in range(nbLacunes):
    u=rd.randint(0,L-1)
    v=rd.randint(0,H-1)
    x0[i] = np.array([u,v])



# Fonction permettant de connaître l'environnement, s'il y a des défauts dans cet environnement, et si oui où
# autresLacunes : ensemble des positions (wrappées, modulo L,H) des AUTRES lacunes (hors la lacune x elle-même)
def environnement(x, autresLacunes, defauts):
    env = [(x[0], (x[1]-1)%H), (x[0], (x[1]+1)%H), ((x[0]-1)%L, x[1]), ((x[0]+1)%L, x[1])]          #on considère les 4 plus proches voisins
    testDef = False
    voisinsDef = []
    testLac = False
    voisinsLac = []
    for a in env:
        if a in defauts:
            testDef = True
            voisinsDef.append(a)
        if a in autresLacunes:
            testLac = True
            voisinsLac.append(a)
    return (env, testDef, voisinsDef, testLac, voisinsLac)

#Fonction donnant la direction dans laquelle est le défaut df par rapport à l'atome x dans l'espace réel (et non réduit avec des conditions aux limites périodiques)
def directionDef(x, df):
    #Les directions sont : 0 gauche, 1 droite, 2 haut, 3 bas
    if (x[0]==0 and df[0]==99)  or (x[1]==0 and df[1]==99)  or (x[0]==99 and df[0]==0)  or (x[1]==99 and df[1]==0):     #x ET le défaut sont sur les bords ==> cas particulier à isoler
        if x[0]==df[0]:
            if x[1]<df[1]:
                return 3
            else:
                return 2
        else:
            if x[0]<df[0]:
                return 0
            else:
                return 1
    else:
        if x[0]==df[0]:     #même colonne
            if x[1]>df[1]:  #x au-dessus du défaut
                return 3    #vers le bas
            else:
                return 2    #vers le haut
            
        else:               #même ligne
            if x[0]>df[0]:  #x à droite du défaut
                return 0    #vers la gauche
            else:
                return 1    #vers la droite

# # Fonction générant une carte de défauts à la concentration donnée
# def mapDefauts(Natomes, c):     
    
#     Ndef = floor(c*Natomes)
#     listeDef = []
    
#     for i in range(Ndef):
#         defautsTrouve=False                  #booleen indiquant si un défaut tiré aléatoirement a été trouvé à une place non occupée précédemment par un autre défaut
#         while defautsTrouve==False:            
#             u = rd.random()
#             v = rd.random()
#             xD = int(u*L)
#             yD = int(v*H)
#             if (xD, yD) not in listeDef:
#                 listeDef.append((xD,yD))
#                 defautsTrouve=True
#     return listeDef

def histogramme_par_x(coords, i, a):
    # Extraire les valeurs de x
    x_values = [x[0] for x in coords]
    
    # Créer une nouvelle figure pour éviter la superposition
    plt.figure()
    
    # Créer l'histogramme
    plt.hist(x_values, bins=50, range=(min(x_values), max(x_values)), edgecolor='black')
    plt.title(f"Histogramme des valeurs de concentration (cumulée selon x) pour i={i} et alpha={a}")
    plt.xlabel("Valeur de concentration selon x")
    plt.ylabel("Fréquence")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"Histo_barre_lacune_N_{N}_i={i}_alpha_{a}.png")

    # Afficher l'histogramme
    plt.show()
    
# Fonction d'animation de l'histogramme des positions des impuretés
def animate(frame_number, bar_container, histogram_data, bins, ax):
    # Effacer les éléments précédents (sauf l'histogramme de base)
    ax.clear()

    # Récupérer les données pour cette frame
    data = histogram_data[frame_number]
    n, _, _ = ax.hist(data, bins, lw=1, ec="mediumturquoise", fc="teal", alpha=0.5)

    # Calculer et tracer la KDE
    kde = gaussian_kde(data)
    x_grid = np.linspace(min(bins), max(bins), 1000)
    ax.plot(x_grid, kde(x_grid) * len(data) * (bins[1] - bins[0]), color="rosybrown", lw=2, label="KDE")

    # Ajouter le texte "i=..."
    ax.text(
        0.02, 0.95,
        f"i = {frame_number * pas_anim}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # Réappliquer les limites et labels
    ax.set_ylim(top=np.max(n) + 5)
    ax.set_xlabel("Position en X")
    ax.set_ylabel("Fréquence")
    ax.set_title("Évolution de la distribution des défauts au fil du temps")
    ax.legend()

    return bar_container.patches






### Boucle faisant des calculs pour chaque couple de (rapport de fréquence défaut/atome du réseau, concentration de défaut)
for b in beta : 
    for a in alpha:
            temps = np.zeros(N+1)       #numpy gardant en mémoire les temps des différents sauts + 0 en référence
            t=0                         #initialisation du temps      
            freqDef = a*freqAt
            freqLacLac = b*freqAt
            
            # listeDefauts=mapDefauts(Natomes, cD)
            listeDefauts = [(x,y) for x in range(L//2,L) for y in range(H) if not np.any(np.all(x0 == np.array([x,y]), axis=1))]
            print("leninit listeDef", len(listeDefauts))
            leninit=len(listeDefauts)
            # listeDefauts = [(x,y) for x in range(L//2,L) for y in range(H) if (x,y) not in x0]     #on supprime les sites occupés par les lacunes
            # listeSuivis = random.sample(listeDefauts, 10)
            
            # Initialisation des positions des lacunes et des défauts
            positions_lacunes = np.zeros((nbLacunes, N + 1, 2), dtype=int)
            for idx, lacune in enumerate(x0):
                positions_lacunes[idx, 0] = lacune
        
            # Chaque trajectoire est une liste de tuples (x, y, t) pour limiter la mémoire occupée
            trajectoires_defauts = [[] for _ in range(len(listeDefauts))]
            for idx, defect in enumerate(listeDefauts):
                trajectoires_defauts[idx].append((defect[0], defect[1], 0.0))  #Positions initiales (à t=0)
        
            Densite = np.zeros((L, H))        
            
            defO = [x for x,y in listeDefauts]
            plt.hist(defO, bins=50, range=(0, L), edgecolor='blue')
            
            # Initialisation des listes pour stocker les données des frames des animations
            histogram_data = []          # Pour l'histogramme
            positions_lacunes_frames = []  # Pour les trajectoires des lacunes
            positions_defauts_frames = []  # Pour les trajectoires des défauts
            positions_reseau_frames = []  # Pour les trajectoires des atomes du réseau
    
            
            
            ### Marche aléatoire
            
            for i in range(1,N):
    
                # Positions courantes de toutes les lacunes
                positions_courantes = [(int(positions_lacunes[k, i-1][0]) % L, int(positions_lacunes[k, i-1][1]) % H) for k in range(nbLacunes)]
                deja_bouge = np.zeros(nbLacunes, dtype=bool) # Indique si une lacune a déjà été déplacée pendant cette itération (suite à un échange avec une autre lacune). Dans ce cas on ne doit pas la refaire bouger à son propre tour.
    
                # Déplacement de chaque lacune
                for lacune_idx in range(nbLacunes):
                    if deja_bouge[lacune_idx]:
                        continue        #cette lacune a déjà bougé ce pas-ci suite à un échange avec une autre lacune
    
                    x = positions_lacunes[lacune_idx, i - 1].copy()
                    u = rd.random()
    
                    autresLacunes = set(positions_courantes[k] for k in range(nbLacunes) if k != lacune_idx)    #set des positions des autres lacunes pour tester l'environnement de la lacune courante (même utilisation que le set "defauts")
                    env, testDef, voisinsDef, testLac, voisinsLac = environnement(x, autresLacunes, set(listeDefauts))
                    
                    ## Calcul des fréquences de saut pour chacun des 4 plus proches voisins
                    
                    #Initialisation des fréquences de saut à freqAt
                    gamma = np.ones(4) * freqAt
                    
                    #Changement des fréquences dans les directions où il y a une impureté ou une lacune
                    if testDef:
                        for df in voisinsDef:                    #Calcul des différentes fréquences de sauts des plus proches voisins
                            dir_def = directionDef(x, df)
                            gamma[dir_def] += (freqDef - freqAt)
    
                    if testLac:
                        for lc in voisinsLac:                    #Interaction avec une lacune voisine : fréquence de saut modifiée
                            dir_lac = directionDef(x, lc)
                            gamma[dir_lac] += (freqLacLac - freqAt)
                    
                    
                    v=rd.random()
                    if v <= 1e-16:                           #On retire les valeurs trop petites qui donneraient un temps d'attente beaucoup trop élevé
                        v = 1e-16
                    t+=-(1/np.sum(gamma))*log(v)
                    temps[i]=t
                    gamma = gamma / np.sum(gamma)            #On normalise gamma pour avoir des segments de probabilités dont l'union forme le segment [0,1]
                    
                    
                    # Détermination de la direction du saut
                    dx, dy = 0, 0
                    if u < gamma[0]:
                        new_pos = ((x[0] - 1) % L, x[1]%H)
                        dx = -1
                    elif u < np.sum(gamma[:2]):
                        new_pos = ((x[0] + 1) % L, x[1]%H)
                        dx = +1
                    elif u < np.sum(gamma[:3]):
                        new_pos = (x[0]%L, (x[1] + 1) % H)
                        dy = +1
                    else:
                        new_pos = (x[0]%L, (x[1] - 1) % H)
                        dy = -1
    
                    new_pos = (int(new_pos[0]), int(new_pos[1]))
    
    
                    # Echange avec une impureté si le site visé en contient une              
                    if testDef and new_pos in set(listeDefauts) and swap == 1:
                        defect_index = listeDefauts.index(new_pos)
                        if len(listeDefauts)!=leninit:
                            print("len avant listeDef", len(listeDefauts), "i:", i, "defect index", defect_index, "lacune n°", lacune_idx)
                        xC, yC, _ = trajectoires_defauts[defect_index][-1]                  #position courante de l'impureté à déplacer
                        nouvelle_position_defaut = ( int(x[0]) % L, int(x[1]) % H ) 
                        trajectoires_defauts[defect_index].append( (nouvelle_position_defaut[0], nouvelle_position_defaut[1], t) )     #l'impureté se déplace dans le sens inverse de la lacune
                        listeDefauts[defect_index] = nouvelle_position_defaut
                        if len(listeDefauts)!=leninit:
                            print("len après listeDef", len(listeDefauts), "i:", i, "lacune n°", lacune_idx)
                        if len(listeDefauts) != len(set(listeDefauts)):
                            print(f"ERRRRRRREUR : doublons dans listeDefauts i={i} pour la lacune n°{lacune_idx} ")
                            print(" ")
                            
                            
                    # Echange direct avec une autre lacune si le site visé en contient une
                    collision_lacune = -1
                    if testLac :
                        for k in range(nbLacunes):
                            if k != lacune_idx and positions_courantes[k] == new_pos:
                                collision_lacune = k                   #indice de la lacune avec laquelle la lacune courante va échanger
                                break
    
                        if collision_lacune >=0:            #i.e si un échange de lacune est à faire,
                            # La lacune cible avance d'un pas dans la direction opposée au déplacement de la lacune courante
                            # x_cible = positions_lacunes[collision_lacune, i - 1].copy()          #position réduite de la lacune avec laquelle la lacune courante va échanger
                            # x_cible[0] = (x_cible[0]-dx)%L
                            # x_cible[1] = (x_cible[1]-dy)%H
                            # positions_lacunes[collision_lacune, i] = x_cible                     #actualisation de la position réelle de la lacune avec laquelle la lacune courante va échanger
                            # positions_courantes[collision_lacune] = (x_cible[0], x_cible[1])
                            # deja_bouge[collision_lacune] = True
                            # Densite[int(x_cible[0]) % L, int(x_cible[1]) % H] += 1
        
                            position_lacune_courante = ( int(x[0]) % L, int(x[1]) % H ) 
                            position_lacune_cible = ( int(positions_lacunes[collision_lacune, i - 1, 0]) % L, int(positions_lacunes[collision_lacune, i - 1, 1]) % H ) # La lacune courante prend la position de la lacune cible 
                            positions_lacunes[lacune_idx, i] = np.array( position_lacune_cible ) # La lacune cible prend la position initiale de la lacune courante 
                            positions_lacunes[collision_lacune, i] = np.array( position_lacune_courante ) # Mise à jour des positions courantes 
                            positions_courantes[lacune_idx] = position_lacune_cible 
                            positions_courantes[collision_lacune] = position_lacune_courante # La lacune cible a déjà été traitée pour ce pas 
                            deja_bouge[collision_lacune] = True
        
        
        
                    # Mise à jour de la position de la lacune courante
                    x[0] = (x[0] + dx)%L
                    x[1] = (x[1] + dy)%H
                    
                    positions_lacunes[lacune_idx, i] = x
                    positions_courantes[lacune_idx] = new_pos
                    Densite[int(x[0]%L), int(x[1]%H)] += 1
                
                # Sauvegarder les positions des défauts tous les pas_anim pas
                if i%(pas_anim) == 0 or (i%10==0 and i<250) or (i%100==0 and i<2500) or (i%1000==0 and i<11000) or (i%5000==0 and i<51000) or i<100:
                    current_defect_positions = [x for x,y in listeDefauts]  # Extraire les positions en x des défauts
                    histogram_data.append(current_defect_positions)
                    # Positions des lacunes et défauts pour les trajectoires
                    positions_lacunes_frames.append([(int(pos[0] % L), int(pos[1] % H)) for pos in positions_courantes])
                    positions_defauts_frames.append([(x, y) for x, y in listeDefauts])
                    positions_reseau_frames.append([(x,y) for x in range(L) for y in range(H) if not np.any(np.all(listeDefauts == np.array([x,y]), axis=1))])
                    print(f"Frame {len(histogram_data)} : i = {i}")
                    temps_frame.append(i)
                    
                    
                # if (i%400==0 and i<2000) or i%(N//10)==0:
                #     histogramme_par_x(autresLacunes, i, a)
                    
            #Marches des impuretés
            # Trajectoires des lacunes
            fig, ax = plt.subplots()
            for idx in range(nbLacunes):
                x_trajectory = [pos[0] for pos in positions_lacunes[idx]]
                y_trajectory = [pos[1] for pos in positions_lacunes[idx]]
                plt.plot(x_trajectory, y_trajectory, marker='o', markersize=2, label=f'Atome {idx + 1}')
            ax.set_xlabel("Position X", fontsize=11)
            ax.set_ylabel("Position Y", fontsize=11)
            ax.set_title(f"Marche aléatoire des {nbLacunes} lacunes avec alpha = {a}", fontsize=11)
            fig.savefig(f'Marche_barre_N_{N}_alpha_{a}_beta_{b}_nbLac_{nbLacunes}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.axis('equal')
            plt.close(fig)
            
            #Trajectoires des impuretés
            fig, ax = plt.subplots()
            for idx in range(min(nbImp, len(trajectoires_defauts))):
                x_trajectory = [pos[0] for pos in trajectoires_defauts[idx]]
                y_trajectory = [pos[1] for pos in trajectoires_defauts[idx]]
                plt.plot(x_trajectory, y_trajectory, marker='o', markersize=2, label=f'Atome {idx + 1}')
            ax.set_xlabel("Position X", fontsize=11)
            ax.set_ylabel("Position Y", fontsize=11)
            ax.set_title(f"Marche aléatoire des {nbImp} impuretés avec alpha = {a}", fontsize=11)
            fig.savefig(f'Marche_barre_N_{N}_alpha_{a}_beta_{b}_nbLac_{nbLacunes}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.axis('equal')
            plt.legend()
            plt.close(fig)
            
            # Sauvegarde des résultats (/!\ fichier .npz de Traj_lacune très volumineux qui N grand (aussi vrai pour Traj_imp dans une moindre mesure))
            # np.savez(f"Traj_lacune_2D_barre_alpha_{a}_beta_{b}_nbLac_{nbLacunes}_N_{N}.npz", positions_lacunes=np.array(positions_lacunes), temps=temps, Densite=Densite)
            
            # with open(f'Traj_imp_2D_barre_alpha_{a}_beta_{b}_nbLac_{nbLacunes}_N_{N}.json', 'w') as f:
            #     json.dump(trajectoires_defauts, f)
            
            
            # ########### Animation pour l'histogramme ###########################
            # # --- Paramètres de l'histogramme ---
            # HIST_BINS = np.linspace(0, L, 50)  # Ajuste selon tes données
            
            # # --- Initialisation de la figure et de l'histogramme ---
            # fig, ax = plt.subplots(figsize=(10, 6))
            # n, _, bar_container = ax.hist(histogram_data[0], HIST_BINS, lw=1, ec="yellow", fc="green", alpha=0.5)
            # ax.set_ylim(top=np.max(n) + 5)
            # ax.set_xlabel("Position en X")
            # ax.set_ylabel("Fréquence")
            # ax.set_title("Évolution de la distribution des défauts au fil du temps")
            
            # # --- Animation ---
            # anim = functools.partial(
            #     animate,
            #     bar_container=bar_container,
            #     histogram_data=histogram_data,
            #     bins=HIST_BINS,
            #     ax=ax
            # )
            
            # ani = animation.FuncAnimation(
            #     fig,
            #     anim,
            #     frames=len(histogram_data),
            #     repeat=False,
            #     blit=False,
            #     interval=200,
            # )
            
            # plt.tight_layout()
            # plt.show()
            
            # ani.save(f"animation_histogramme_kde_N_{N}.gif", writer="ffmpeg", fps=5, dpi=200)
            
            
            ################ ANIMATION DE LA MARCHE ################################################################### 
             
    
    
             # nbAtomes, nbPas, _ = pos_atomes.shape
    
             # fig, ax = plt.subplots(figsize=(8, 8))
             
             # xmin = min(pos_atomes[:, :, 0].min(), positions[:, 0].min())
             # xmax = max(pos_atomes[:, :, 0].max(), positions[:, 0].max())
             # ymin = min(pos_atomes[:, :, 1].min(), positions[:, 1].min())
             # ymax = max(pos_atomes[:, :, 1].max(), positions[:, 1].max())
             
             # ax.set_xlim(xmin, xmax)
             # ax.set_ylim(ymin, ymax)
             # ax.set_xlabel("x")
             # ax.set_ylabel("y")
             # ax.set_title("Vacancy-mediated diffusion in 2D cubic lattice")
             # ax.grid(True)
             
             # # --- scatter des atomes ---
             # scat_atomes = ax.scatter(
             #     pos_atomes[:, 0, 0],
             #     pos_atomes[:, 0, 1],
             #     s=20,
             #     c='black',
             #     label='Tracer atom'
             # )
             
             # # --- scatter de la lacune / défaut ---
             # scat_defaut = ax.scatter(
             #     positions[0, 0],
             #     positions[0, 1],
             #     s=80,
             #     c='blue',
             #     marker='x',
             #     label='Vacancy'
             # )
             
             # ax.legend(fontsize = 10)
             
             # def update(frame):
             #     scat_atomes.set_offsets(pos_atomes[:, frame, :])
             #     scat_defaut.set_offsets(positions[frame])
             #     ax.set_title(f"Time step : {frame}")
             #     return scat_atomes, scat_defaut
             
             # ani = FuncAnimation(
             #     fig,
             #     update,
             #     frames=range(0, nbPas, 20),  # 3ème argument = taille des sauts dans les frames
             #     interval=50,  #durée du frame
             #     blit=False
             # )
             
             # ani.save(
             #     "diffusion_lacunes_barre.gif",
             #     writer="pillow",
             #     dpi=100
             # )
             
             # #plt.show()
            
            # Paramètres pour les animations
            frames = len(histogram_data)  # Nombre de frames
            interval = 200  # Intervalle entre les frames (en ms)
            dpi = 100  # Résolution

            # --- 1. Animation de l'histogramme ---
            def create_histogram_animation():
                fig, ax = plt.subplots(figsize=(10, 6))
                HIST_BINS = np.linspace(0, L, 50)
                ax.set_ylim(0, borneMaxHisto)

                def update_hist(frame):
                    ax.clear()
                    data = histogram_data[frame]
                    n, _, _ = ax.hist(data, HIST_BINS, lw=1, ec="mediumturquoise", fc="teal", alpha=0.5)

                    # Ajouter la KDE
                    kde = gaussian_kde(data)
                    x_grid = np.linspace(min(HIST_BINS), max(HIST_BINS), 1000)
                    ax.plot(x_grid, kde(x_grid) * len(data) * (HIST_BINS[1] - HIST_BINS[0]), color="rosybrown", lw=2, label="Profil lissé de concentration")

                    # ax.set_ylim(20)
                    ax.set_xlabel("Concentration cumulée selon X")
                    ax.set_ylabel("Fréquence")
                    ax.set_title(f"Distribution des impuretés (pas n° {temps_frame[frame]})")  # <-- Utilise `pas_anim` au lieu de `N//20`
                    ax.legend(loc='upper left')
                    return ax.patches

                ani_hist = animation.FuncAnimation(
                    fig,
                    update_hist,
                    frames=frames,
                    interval=interval,
                    blit=False,
                    repeat=False
                )
                ani_hist.save(f"animation_histogramme_N_{N}_alpha_{a}_beta_{b}_nbLacunes_{nbLacunes}.gif", writer="pillow", dpi=dpi)
                plt.close(fig)
                return ani_hist

            # --- 2. Animation des trajectoires ---
            def create_trajectories_animation():
                fig, ax = plt.subplots(figsize=(25, 5))
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_xlim(0-1/2,L-1/2)
                ax.set_ylim(0-1/2,H-1/2)
                ax.set_aspect('equal')
                ax.set_title("Trajectoires des lacunes et défauts")
                ax.grid(False)

                # Initialiser les scatters
                scat_reseau = ax.scatter(
                    [pos[0] for pos in positions_reseau_frames[0]],
                    [pos[1] for pos in positions_reseau_frames[0]], 
                    s=80,
                    c='lightsteelblue',
                    marker='o',
                    alpha=0.5,
                    label='Atomes du réseau'
                ) 
                scat_lacunes = ax.scatter(
                    [pos[0] for pos in positions_lacunes_frames[0]],
                    [pos[1] for pos in positions_lacunes_frames[0]],
                    s=80,
                    c='darkred',
                    marker='s',
                    label='Lacunes'
                )
                scat_defauts = ax.scatter(
                    [pos[0] for pos in positions_defauts_frames[0]],
                    [pos[1] for pos in positions_defauts_frames[0]], 
                    s=80,
                    c='teal',
                    marker='o',
                    label='Impuretés'
                )
                ax.legend(loc='upper left', fontsize=11)

                def update_traj(frame):
                    
                    # Mettre à jour les positions des atomes du réseau
                    reseau_x = [pos[0] for pos in positions_reseau_frames[frame]]
                    reseau_y = [pos[1] for pos in positions_reseau_frames[frame]]
                    scat_reseau.set_offsets(list(zip(reseau_x, reseau_y)))
                    
                    
                    # Mettre à jour les positions des lacunes
                    lacunes_x = [pos[0] for pos in positions_lacunes_frames[frame]]
                    lacunes_y = [pos[1] for pos in positions_lacunes_frames[frame]]
                    scat_lacunes.set_offsets(list(zip(lacunes_x, lacunes_y)))

                    # Mettre à jour les positions des défauts
                    defauts_x = [pos[0] for pos in positions_defauts_frames[frame]]
                    defauts_y = [pos[1] for pos in positions_defauts_frames[frame]]
                    scat_defauts.set_offsets(list(zip(defauts_x, defauts_y)))

                    ax.set_title(f"Trajectoires (pas n° {temps_frame[frame]})") 
                    return scat_lacunes, scat_defauts

                ani_traj = animation.FuncAnimation(
                    fig,
                    update_traj,
                    frames=frames,
                    interval=interval,
                    blit=False,
                    repeat=False
                )
                ani_traj.save(f"animation_trajectoires_N_{N}_alpha_{a}_beta_{b}_nbLacunes_{nbLacunes}.gif", writer="pillow", dpi=dpi)
                plt.close(fig)
                return ani_traj

            # --- 3. Animation combinée (trajectoires + histogramme) ---
            def create_combined_animation():
                fig, (ax_traj, ax_hist) = plt.subplots(2, 1, figsize=(50, 30))

                # --- Sous-graphe des trajectoires ---
                ax_traj.set_xlim(0-1/2, L-1/2)
                ax_traj.set_ylim(0-1/2, H-1/2)
                ax_traj.set_xlabel("x")
                ax_traj.set_ylabel("y")
                ax.set_aspect('equal')
                ax_traj.set_title("Trajectoires des lacunes et impuretés", fontsize=18)
                ax_traj.grid(False)

                scat_reseau = ax_traj.scatter(
                    [pos[0] for pos in positions_reseau_frames[0]],
                    [pos[1] for pos in positions_reseau_frames[0]], 
                    s=80,
                    c='lightsteelblue',
                    alpha=0.5,
                    marker='o',
                    label='Atomes du réseau'
                )    

                scat_lacunes = ax_traj.scatter(
                    [pos[0] for pos in positions_lacunes_frames[0]],
                    [pos[1] for pos in positions_lacunes_frames[0]],
                    s=80,
                    c='darkred',
                    marker='s',
                    label='Lacunes'
                )
                scat_defauts = ax_traj.scatter(
                    [pos[0] for pos in positions_defauts_frames[0]],  # <-- Utilise `positions_defauts_frames`
                    [pos[1] for pos in positions_defauts_frames[0]],  # <-- Utilise `positions_defauts_frames`
                    s=80,
                    c='teal',
                    marker='o',
                    label='Impuretés'
                )
                
                ax_traj.legend(loc='upper left')
                ax_traj.set_aspect('equal')

                # --- Sous-graphe de l'histogramme ---
                HIST_BINS = np.linspace(0, L, 50)
                n, _, _ = ax_hist.hist(histogram_data[0], HIST_BINS, lw=1, ec="mediumturquoise", fc="teal", alpha=0.5)
                ax_hist.set_ylim(0, borneMaxHisto)
                ax_hist.set_xlim(0, L)

                def update_combined(frame):
                    # Mettre à jour les trajectoires
                    lacunes_x = [pos[0] for pos in positions_lacunes_frames[frame]]
                    lacunes_y = [pos[1] for pos in positions_lacunes_frames[frame]]
                    scat_lacunes.set_offsets(list(zip(lacunes_x, lacunes_y)))

                    defauts_x = [pos[0] for pos in positions_defauts_frames[frame]]
                    defauts_y = [pos[1] for pos in positions_defauts_frames[frame]] 
                    scat_defauts.set_offsets(list(zip(defauts_x, defauts_y)))
                    
                    reseau_x = [pos[0] for pos in positions_reseau_frames[frame]]
                    reseau_y = [pos[1] for pos in positions_reseau_frames[frame]]
                    scat_reseau.set_offsets(list(zip(reseau_x, reseau_y)))

                    ax_traj.set_title(f"Trajectoires (i = {temps_frame[frame]})", fontsize=20)

                    # Mettre à jour l'histogramme
                    ax_hist.clear()
                    data = histogram_data[frame]
                    n, _, _ = ax_hist.hist(data, HIST_BINS, lw=1, ec="mediumturquoise", fc="teal", alpha=0.5)

                    kde = gaussian_kde(data)
                    x_grid = np.linspace(min(HIST_BINS), max(HIST_BINS), 1000)
                    ax_hist.plot(x_grid, kde(x_grid) * len(data) * (HIST_BINS[1] - HIST_BINS[0]), color="rosybrown", lw=2, label="KDE")

                    ax_hist.set_xlabel("Concentration cumulée selon X")
                    ax_hist.set_ylabel("Fréquence")
                    ax_hist.set_title(f"Distribution des impuretés (pas n° {temps_frame[frame]})", fontsize=20)
                    ax_hist.legend(loc='upper left')

                    return scat_lacunes, scat_defauts, ax_hist.patches

                ani_combined = animation.FuncAnimation(
                    fig,
                    update_combined,
                    frames=frames,
                    interval=interval,
                    blit=False,
                    repeat=False
                )
                ani_combined.save(f"animation_combined_N_{N}_alpha_{a}_beta_{b}_nbLacunes_{nbLacunes}.gif", writer="pillow", dpi=dpi)
                plt.close(fig)
                return ani_combined
            
            # --- Générer les 3 animations ---
            print("Génération des animations...")
            ani_hist = create_histogram_animation()
            ani_traj = create_trajectories_animation()
            ani_combined = create_combined_animation()
            print("Animations sauvegardées !")
