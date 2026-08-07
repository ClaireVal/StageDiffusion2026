import numpy.random as rd
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import exp, floor, log
from ase.units import kB
import numpy as np

#mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#mpl.rc('text', usetex=True)
mpl.rcParams['ytick.labelsize']=14
mpl.rcParams['xtick.labelsize']=14
#mpl.rcParams['text.fontsize']=20

mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['figure.figsize'] = (12, 12)

plt.close("all")
# Taille de l'espace considéré
H = 100
L = 100
Natomes = H*L
N=100000            #nb de pas

swap=0             #activation du swap si 1, désactivation si 0
tpsVar=1           #si =1 temps variable (et si =0 tps constant, mais code uniquement pour temps variable)

### Paramètres du système
# # L'ordre de grandeur pour les énergies de formation et de migration d'un défaut sont de l'ordre de 1eV (cf rapport de stage de Manon Dewynter)
G_M = 1.5 #eV = barrière d'énergie pour l'échange lacune / atome du réseau
# G_Mdef = 0.5 #eV = barrière d'énergie pour l'échange lacune / atome substitué
# G_F = 1.0 #eV = énergie libre de Gibbs de formation du défaut substitutionnel
T = 300 #K
nu_0 = 10e13

alpha = [1]
freqAt = nu_0 * exp(-G_M/(kB*T))               #Fréquence de saut des atomes du réseau

### Positions aléatoires des défauts
# cDef = exp(-G_F/(kB*T))
# Ndef = floor(cDef*N)
# Ndef = 10
# cDef = Ndef/Natomes
cDef = [0]



x0 = [50,50]



# Fonction permettant de connaître l'environnement, s'il y a des défauts dans cet environnement, et si oui où
def environnement(x):
    env = [(x[0], (x[1]-1)%H), (x[0], (x[1]+1)%H), ((x[0]-1)%L, x[1]), ((x[0]+1)%L, x[1])]          #on considère les 4 plus proches voisins
    testDef = False
    voisinsDef = []
    for a in env:
        if a in defauts:
            testDef = True
            voisinsDef.append(a)
    return (env, testDef, voisinsDef)

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

# Fonction générant une carte de défauts à la concentration donnée
def mapDefauts(Natomes, c):     
    
    Ndef = floor(c*Natomes)
    listeDef = []
    
    for i in range(Ndef):
        defautsTrouve=False                  #booleen indiquant si un défaut tiré aléatoirement a été trouvé à une place non occupée précédemment par un autre défaut
        while defautsTrouve==False:            
            u = rd.random()
            v = rd.random()
            xD = int(u*L)
            yD = int(v*H)
            if (xD, yD) not in listeDef:
                listeDef.append((xD,yD))
                defautsTrouve=True
    return listeDef


### Boucle faisant des calculs pour chaque couple de (rapport de fréquence défaut/atome du réseau, concentration de défaut)
for a in alpha:
    for cD in cDef:
        temps = np.zeros(N+1)       #numpy gardant en mémoire les temps des différents sauts + 0 en référence
        t=0                         #initialisation du temps      
        freqDef = a*freqAt
        
        listeDefauts=mapDefauts(Natomes, cD)
        defauts = set(listeDefauts)
        
        ### Initialisation des pas
        gauche = 0
        droite = 0
        haut = 0
        bas = 0
    
        Densite = np.zeros((L, H))
        positions = np.zeros((N+1,2))
        x=x0.copy()
        pos_reelle = np.zeros((N+1,2))
        pos_reelle[0] = x0
        positions[0] = x0               

        
        ### Marche aléatoire
        
        for i in range(1,N+1):
            u = rd.random()
            env = environnement(x)
            
            ## Calcul des fréquences de saut pour chacun des 4 plus proches voisins
            gamma = np.ones(4) * freqAt
            
            if env[1]:
                envDef = env[2]
                
                for df in envDef:                    #Calcul des différentes fréquences de sauts des plus proches voisins
                    dir_def = directionDef(x, df)
                    gamma[dir_def] += (freqDef - freqAt)
            
            v=rd.random()
            if v <= 1e-16:
                v = 1e-16
            t+=-(1/np.sum(gamma))*log(v)
            temps[i]=t
            gamma = gamma / np.sum(gamma)            #On normalise gamma pour avoir des segments de probabilités dont l'union forme le segment [0,1]
            
            
            ## Déplacement de la lacune x 
            dx = 0
            dy = 0
                    
            if u < gamma[0]:
                if ((x[0]-1, x[1]) in defauts) and swap==1:
                    defauts.remove((x[0]-1, x[1]))
                    defauts.add((x[0], x[1]))
                dx = -1                              
                
            elif u < np.sum(gamma[:2]):
                if ((x[0]+1, x[1]) in defauts) and swap==1:
                    defauts.remove((x[0]+1, x[1]))
                    defauts.add((x[0], x[1]))
                dx = +1                                
                    
            elif u < np.sum(gamma[:3]):
                if ((x[0], x[1]+1) in defauts) and swap==1:
                    defauts.remove((x[0], x[1]+1))
                    defauts.add((x[0], x[1]))
                dy = +1
                    
            else:
                if ((x[0], x[1]-1) in defauts) and swap==1:
                    defauts.remove((x[0], x[1]-1))
                    defauts.add((x[0], x[1]))
                dy = -1
            
            
            # position non périodique (réelle) pour le calcul du MSD
            pos_reelle[i] = pos_reelle[i-1] + np.array([dx, dy])
                
            # position réduite dans la boîte L*H et le mapping de densité de passage
            x[0] = (x[0] + dx) % L
            x[1] = (x[1] + dy) % H
            positions[i] = x.copy()
            Densite[x[0], x[1]]+=1
            
            
    
                
        # fig, ax = plt.subplots(2)
        # abscisses = [elem[0] for elem in positions]
        # ordonnees = [elem[1] for elem in positions]
        
        # ax[0].scatter(abscisses, ordonnees, c=range(len(abscisses)), cmap='viridis')
        # ax[0].set_title("Marche aléatoire 2D", fontsize=12)
        # ax[0].set_xlabel("Position horizontale", fontsize=11)
        # ax[0].set_ylabel("Position verticale", fontsize=11)
        # ax[0].tick_params(axis='both', labelsize=14)
        # fig.suptitle(f"Evolution du coefficient de diffusion et du facteur de corrélation en fonction \n du temps d'observation tau pour une trajectoire unique stockée en mémoire\n cDef={cD}, alpha={a} et N={N}", fontsize=16)
        # ax[0].plot(dtLogListe_C, DLog_Cour, 'o',color = "red")
        # ax[0].set_xlabel("tau", fontsize=14)
        # ax[0].set_ylabel("D", fontsize=14)
        # ax[0].set_title(f"D de la longue trajectoire en fonction de tau avec cDef = {cD} et alpha = {a}", fontsize=14)
        # ax[0].set_xscale("log")
        # ax[0].set_yscale("log")
        # ax[1].plot(dtLogListe_C, fLog_Cour, 'o',color = "blue")
        # ax[1].set_xlabel("tau", fontsize=14)
        # ax[1].set_ylabel("D", fontsize=14)
        # ax[1].set_title(f"f de la longue trajectoire en fonction de tau avec cDef = {cD} et alpha = {a}", fontsize=14)
        # ax[1].set_xscale("log")
        # ax[1].set_yscale("log")
        # fig.subplots_adjust(hspace = 0.5)     
        
        # fig, ax = plt.subplots()
        # ax.plot(dtLogListe_C, MSDLog_Cour, 'o', color="black")
        # ax.set_xlabel("tau", fontsize=14)
        # yFitL = exp(poptC4[0])*(dtLogListe_C**poptC4[1])
        # ax.plot(dtLogListe_C, yFitL, label="Courbe fittée trajectoire longue", color="orange",ls='--',lw=2)
        # ax.set_ylabel("MSD", fontsize=14)
        # ax.set_title(f"MSD de la longue trajectoire en fonction de tau avec cDef = {cD} et alpha = {a}", fontsize=14)
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        
        # fig, ax = plt.subplots()
        # abscisses = [elem[0] for elem in pos_reelle.reshape(N+1, 2)]
        # ordonnees = [elem[1] for elem in pos_reelle.reshape(N+1, 2)]
        # ax.scatter(abscisses, ordonnees, c=range(len(abscisses)), cmap='viridis')
        # ax.set_xlabel("positions X", fontsize=14)
        # ax.set_ylabel("positions Y", fontsize=14)
        # ax.set_title(f"Marche aléatoire avec cDef = {cD} et alpha = {a}", fontsize=14)
            
        # fig, ax = plt.subplots()
        # im = ax.imshow(Densite.T, cmap='plasma', origin='lower')
        # fig.colorbar(im, ax=ax)
        # ax.set_xlabel('X', fontsize=14)
        # ax.set_ylabel('Y', fontsize=14)
        
        # ax.set_title(f"Mapping de la densité de passage pour alpha ={a}, concentration de défauts = {cD}\n et position initiale {x0}", fontsize=14)
        # ax.tick_params(axis='both', labelsize=14)
        
        # XDisplay = []
        # YDisplay = []
        # for (aDef,bDef) in listeDefauts:
        #     XDisplay.append(aDef)
        #     YDisplay.append(bDef)
            
        # XDefCourant = []
        # YDefCourant = []
        # for (aD,bD) in defauts:
        #     XDefCourant.append(aD)
        #     YDefCourant.append(bD)
        
        # plt.plot(XDisplay, YDisplay, "^r", label='Positions initiales défauts', alpha=0.5)
        # plt.plot(XDefCourant, YDefCourant, "^w", label='Positions finales défauts', alpha=0.5)
        # plt.legend(prop={'size':12})
        # plt.show()                         
        
        np.savez(f"Traj_swap_{swap}_tpsVariable_{tpsVar}_N_{N}_cD_{cD}_alpha_{a}.npz", pos_reelle=pos_reelle, temps=temps, Densite=Densite, positions=positions, listeDefauts=listeDefauts, defautsFin=defauts)
