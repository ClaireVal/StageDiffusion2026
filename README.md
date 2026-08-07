# StageDiffusion2026
Répertoire permettant de sauvegarder les codes liés au stage "Etude de diffusion de défauts dans des alliages par calculs ab initio".
Dans ce dossier, il y a l'ensemble des étapes du développement du code, de briques de base (calculs de MSD) aux marches 2D/3D.

## I - Marches aléatoires

Ceci constitue le dossier principal. Il est à explorer de manière incrémentale en commençant par la marche 1D. 

### "1D" : 
Avec ce modèle-jouet, un premier code de marche 1D très simple a été écrit, et le MSD a été calculé de différentes manières: échelle linéaire vs log, et trajectoire longue unique vs multiples trajectoires courtes pour vérifier l'hypothèse d'ergodicité. On notera que les trajectoires et calculs de MSD peuvent être générés et sauvegardés dans un fichier "MSD.npz" par le code "Marche aléatoire 1D avec MSD" puis analysées dans un second temps par d'autres codes.

### "2D - Pas de temps constants" : 
Ensuite, passons à la marche 2D à pas de temps constants. On retrouve le schéma de génération de trajectoires et sauvegarde. Les codes de marche permettent de simuler des trajectoires, qui sont ensuite stockées au format .npz. Les codes d'analyse permettent de calculer le MSD, le coefficient de diffusion D et le facteur de corrélation f à partir des fichiers .npz (attention, il faut bien renseigner les bons paramètres au début du code d'analyse afin d'importer le bon fichier de sauvegarde, et le fichier de sauvegarde doit être dans le même dossier que le code d'analyse). Des codes avec et sans swap des impuretés sont aussi écrits. Ces codes sont écrits pour des milieux inhomogènes où il y aurait des atomes du réseau (ayant une fréquence freqAt d'échange avec la lacune) et des impuretés à la concentration cDef (ayant une fréquence alpha*freqAt d'échange avec la lacune). Une étude paramétrique de D ou f selon cDef et alpha peut être menée grâce à ces codes (cf dossier "Recherche loi d'échelle"). L'influence de plusieurs paramètres a été testée dans "Tests de fit" : détermination d'une éventuelle plage de fit optimale et le tracé de la pente du MSD et de D selon une borne minimale du fit dt_min (pour voir si cela convergait en s'affranchissant du régime propre aux petits dt).
Dans ces premiers codes, on s'était intéressés au déplacement des lacunes. Cependant, des marches 2D de suivi d'atomes du réseau sont aussi écrites dans le dossier "Cas tests Point de vue atomes - 2D cubique et nid d'abeilles". Les structures étudiées sont des structures cubiques 2D et nid d'abeille. Le MSD est ici calculé en se plaçant du point de vue des atomes du réseau et non du point de vue des lacunes. Le code de variabilité permet de donner un intervalle de variabilité en cas de multiples mesures de D ou f pour un même jeu de paramètre sur une structure donnée.

### "2D - Pas de temps variables" : 
Jusqu'à maintenant, les temps d'attente entre deux sauts étaient constants et étaient associés à un pas de la simulation. Néanmoins, le temps d'attente peut être calculé à partir des fréquences de saut (cf. TET) donc ils peuvent être calculés à la volée via t = - ln(u)/(somme freq voisins) avec u une variable aléatoire tirée uniformément sur [0,1]. 
La structure de génération de trajectoire reste donc la même, mais une liste des temps est donc gardée en mémoire.
Par ailleurs, le code de calcul du MSD est changé afin de prendre en compte la variabilité des temps d'attente. Une méthode de binning est employée afin de moyenner des sauts ayant des sauts d'attente proches.

### "2D - Plusieurs lacunes" : 
Une étude sur une barre 2D avec plusieurs lacunes et des impuretés a aussi été menée. Des animations de trajectoires et d'histogrammes de concentration sont générées et permettent de suivre, au fil des pas des lacunes, la distribution des impuretés et des lacunes. Un code d'analyse calcule les MSD,D et f des impuretés. Etant donnés la longueur de la simulation et le nombre important de traceurs, une structure alternative de sauvegarde des trajectoires a été choisie. La trajectoire est stockée sous forme d'une liste de triplets (x_i, y_i, t_i) avec i les instants où la particule a sauté. Ainsi, s'il y a N pas de simulation mais que la particule n'a sauté que m fois, alors sa trajectoire ne comporte que m triplets. Cela allège grandement la sauvegarde.
Un bug est à noter : il semblerait que, rarement, un défaut soit dédoublé dans la liste des défauts listeDefauts. L'hypothèse privilégiée est une erreur de coordonnées réduites avec une impureté qui se placerait en dehors de la boîte mais rien de précis ni sûr n'est déterminé pour le moment.

### "3D" : 
Code simple reprenant le principe des codes de marche aléatoire 2D à pas de temps constants et avec un paysage d'impuretés et une lacune, mais l'appliquant à 3 dimensions.

### "Début algo paramétrable" : 
Ces codes sont plutôt inaboutis. Il s'agit d'un code qui génererait la trajectoire d'une lacune dans un milieu complexe, décrit à partir de quelques variables d'entrée simples : liste des plus proches voisins (ppv), liste de description des impuretés du milieu (inputDef) et la fréquence d'échange lacune/atome du réseau (freqAt).



## II - Tests

### A - Tests optimisation Pybind
La fonction de calcul du MSD est clairement l'étape la plus lente dans l'ensemble des codes Marche et Analyse. Ainsi, nous avons cherché une méthode pour l'optimiser. Il a donc été décidé de passer cette fonction en C++ car python n'est pas du tout optimisé pour les boucles for alors que le C++ beaucoup plus. Il s'agit donc ici d'écrire la fonction MSD en C++ dans un fichier .cpp, de faire un fichier setup.py pour le convertir en module importable sur python, et d'exécuter le fichier setup dans le terminal via la ligne de code "python setup.py install (--user)". Cela crée un module .egg-info dans le même dossier, que l'on peut importer dans python. Toute cette démarche utilise la bibliothèque Pybind11. Attention, il faut bien utiliser la même version de python pour compiler le module et pour l'exécuter. 
On retrouve deux dossiers dans Tests_optimisation_Pybind:
1. Exemple_pybind : Exemple très simple pour prendre en main Pybind et comprendre le principe
2. Marches_pybind : Application aux marches aléatoires. Dans un cas, le MSD a été simplement traduit en C++ (no openmp) et dans le second, les boucles ont été parallélisées grâce à OpenMP (openmp).

### B - Tests sous-échantillonage

Cela correspond à une démarche inaboutie par manque de temps. La contrainte en 0(N²) sur le temps de calcul est très mauvaise lorsque N augmente grandement. Une idée afin de diminuer la complexité était de sous-échantilloner les pas de temps pour lequel le MSD était calculé, en ne prenant en compte que certains sauts. Des tests pour des sous-échantillonage des pas de temps de début ont été testés et les résultats consignés dans "Résultats sous-échantillonage.ods" mais cette démarche nécessite d'être largement approfondie pour avoir des résultats fiables et une formule du seuil critique d'échantillonage nécessaire en fonction de N.



## III - Biblio

L'étude bibliographique est loin d'être exhaustive, mais certains papiers sont tout de même fondateurs pour mon stage, tels que:
- 2007_Book_DiffusionInSolids
- Algorithmes > introduction_to_kinetic_monte_carlo_Voter
- Rapport de stage de Manon Dewynter et documents de Romuald et Alizée sur la diffusion et le facteur de corrélation (dossier Rapport de stage et documents Romuald&Alizée)



## IV - Fonctions annexes

On retrouve deux dossiers dans ce dossier :
### 1. "Annexes" : 
Ce sont des codes écrits en début de stage qui n'ont finalement pas servi. Ils permettent de retrouver une distribution de probabilité usuelle à partir d'une cumulée.
### 2. "Codes_init" : 
Ces codes sont les codes initiaux reçus au début de mon stage. Ils ont posé certains bases qui m'ont servies pendant mon stage, notamment pour les calculs de MSD.


En espérant que ce readme aura clarifié la structure de cette sauvegarde, bon courage à celleux travaillant sur ce projet !!!
Un petit pas pour la lacune, un grand pas pour l'humanité...! :)
