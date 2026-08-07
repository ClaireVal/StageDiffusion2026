import numpy.random as rd
import matplotlib.pyplot as plt
from math import exp, floor, log
from ase.units import kB
import numpy as np
import pickle

plt.close("all")


# Géométrie du système
X=15
Y=15
Z=15
Natomes = X*Y*Z
N=1000000          #nb de pas

swap=1             #activation du swap si 1, désactivation si 0
x0 = np.array([0,0,0], dtype='float64')


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

# Caractérisation des impuretés mises en jeu par des couples (freq_i, concentration_i):
inputDef = np.array([[freqAt*1000, 0.001], [freqAt, 0.001], [freqAt*0.001, 0.001]]) 
if np.sum(inputDef[:, 1])>1:
    print("Concentration totale en défauts supérieure à 100%")

freqDef = inputDef[:,0]
cDef = inputDef[:,1]
alpha = freqDef/freqAt

# Fonction générant une carte de défauts à la concentration donnée
def mapDefautsModul(Natomes, listeCDef):
    Ntot=floor(Natomes*np.sum(listeCDef))
    listeDef = np.empty((Ntot,dimension), dtype="float64")
    labelsDef = np.zeros(Ntot, dtype="int64")
    nDefTrouve = 0                                #Pointeur de remplissage des arrays listeDef et labelsDef
    for i, cD in enumerate(listeCDef):      
        Ndef_i = floor(cD*Natomes)                #nb de défauts à la fréquence de saut freqDef[i]
        for k in range(Ndef_i):
            defautsTrouve=False                   #booleen indiquant si un défaut tiré aléatoirement a été trouvé à une place non occupée précédemment par un autre défaut
            while defautsTrouve==False:            
                u = rd.random()
                v = rd.random()
                w = rd.random()
                xD = int(u*X)
                yD = int(v*Y)
                zD = int(w*Z)
                if not np.any(np.all(listeDef == (xD, yD, zD), axis=1)):
                    listeDef[nDefTrouve] = (xD,yD,zD)
                    labelsDef[nDefTrouve] = i
                    defautsTrouve=True
                    nDefTrouve+=1
    return listeDef, labelsDef



# Fonction permettant de connaître l'environnement selon un pattern de plus proche voisin renseigné en argument
def environnementModul(ppv, x):
    pos_defauts= np.array([valeur[1][-1][1] for valeur in defauts.values()])
    #ppv renseigne les plus proches voisins sous la forme de couple (dx, dy)
    env = np.empty((Nppv,dimension))
    indiceEnv = np.full((Nppv), -1 )        #array associant à chaque élément de l'environnement -1 s'il ne s'agit pas d'un défaut, ou l'indice du défaut dans la liste defauts s'il s'agit d'un défaut. On l'initialise à -1 partout au début.
    for k in range(Nppv):
        dx,dy,dz = ppv[k]
        env[k] = (x[0]+dx, x[1]+dy, x[2]+dz)        #On stocke l'élément de l'environnement en coordonées non réduite pour pouvoir en extraire les dx, dy, dz par la suite
    freqEnv = np.empty(Nppv)
    for i, a in enumerate(env):
        xA, yA, zA = a
        a_reduit = np.asarray([xA%X, yA%Y, zA%Z], dtype=pos_defauts.dtype)        #Mais il faut repasser en coordonnées réduites pour déterminer si l'élément est un défaut ou non (car defauts est une liste de positions en coordonnées réduites)
        if len(np.where(np.all(pos_defauts == a_reduit, axis=1))[0])>0 and np.any(np.where(np.all(pos_defauts == a_reduit, axis=1))[0].item()):
            indice = int(np.argwhere(np.all(pos_defauts == a_reduit, axis=1))[0].item())
            indiceEnv[i] = indice
            freqEnv[i] = defauts[indice][0]
        else:
            freqEnv[i] = freqAt
            
    return (env, freqEnv, indiceEnv)


def deplacement(environ, gammaCumul, indiceEnviron, xLacune, instant):
    pos_defauts= np.array([valeur[1][-1][1] for valeur in defauts.values()])
    ## Déplacement de la lacune, positiée en x à ce pas-là 
    dx = 0
    dy = 0
    dz = 0
    u = rd.random() 
    ind=0
    while u > gammaCumul[ind]:
        ind+=1
    dx, dy, dz = environ[ind]-x
    envX, envY, envZ = environ[ind]
    
    #Swap : Actualisation de la position des défauts. Le test indiceEnviron[ind]>=0 permet de déterminer s'il s'agit d'un défaut ou non (cf environnement et initialisation à -1)
    if  indiceEnviron[ind]>=0 and np.any(np.all(pos_defauts == (envX%X, envY%Y, envZ%Z), axis=1)) and swap==1:
        pos_defauts= np.array([valeur[1][-1][1] for valeur in defauts.values()])
        print("positions defauts courantes - Avant", pos_defauts, indiceEnviron[ind])
        #On ajoute [instant, nouvelle position] dans defauts[indiceEnviron][1]
        defauts[indiceEnviron[ind]][1].append((instant, xLacune))
        pos_defauts= np.array([valeur[1][-1][1] for valeur in defauts.values()])
        print("positions defauts courantes - Après", pos_defauts)
        
    return dx, dy, dz
#-----------------------------------------------------------------------------------------------------------------------------------------------------------


temps = np.zeros(N+1)       #numpy gardant en mémoire les temps des différents sauts + 0 en référence
t=0                         #initialisation du temps      

listeDefauts, labelsDefauts=mapDefautsModul(Natomes, cDef)
#Dictionnaire stockant, pour chaque défaut i, sa fréquence de saut par rapport à la lacune, et une liste des [instant de début de présence à la position i, position i]
defauts = {}
for i, (pos_init, label) in enumerate(zip(listeDefauts, labelsDefauts)):
    positionInit = np.array(pos_init)
    defauts[i] = [freqDef[label], [(0, positionInit)]]

print("INITIALEMENT", defauts)
Densite = np.zeros((X,Y,Z))
positions = np.zeros((N+1,dimension))
pos_reelle = np.zeros((N+1,dimension))
pos_reelle[0] = x0
positions[0] = x0               
x=x0.copy()

### Marche aléatoire
for i in range(1,N+1):    
    envPPV, gamma, indicePPV = environnementModul(ppv, x)

    v=rd.random()
    if v <= 1e-16:
        v = 1e-16
    t+=-(1/np.sum(gamma))*log(v)
    temps[i]=t
    gamma = gamma / np.sum(gamma)            #On normalise gamma pour avoir des segments de probabilités dont l'union forme le segment [0,1]
    
    gammaCumul = np.array([np.sum(gamma[:j]) for j in range(1, Nppv+1)])
    dx, dy, dz = deplacement(envPPV, gammaCumul, indicePPV, x, t)
    
    
    # position non périodique (réelle) pour le calcul du MSD
    pos_reelle[i] = pos_reelle[i-1] + np.array([dx, dy, dz])
        
    # position réduite dans la boîte L*H et le mapping de densité de passage
    x[0] = float((x[0] + dx) % X)
    x[1] = float((x[1] + dy) % Y)
    x[2] = float((x[2] + dz) % Z)
    positions[i] = x.copy()
    Densite[int(x[0]),int(x[1]),int(x[2])]+=1

#Sauvegarde
with open("defauts_SuiviDef_tpsVariable_N_{N}_ppv_{ppv}_cDef_{cDef}_freqDef_{freqDef}_swap_{swap}_XYZ_{X}{Y}{Z}.pkl", "wb") as f:
    pickle.dump(defauts, f)
np.savez(f"Traj_SuiviDef_tpsVariable_N_{N}_ppv_{ppv}_cDef_{cDef}_freqDef_{freqDef}_swap_{swap}_XYZ_{X}{Y}{Z}.npz", pos_reelle=pos_reelle, temps=temps, Densite=Densite, positions=positions, listeDefauts=listeDefauts, labelsDefauts=labelsDefauts, defauts=defauts)
