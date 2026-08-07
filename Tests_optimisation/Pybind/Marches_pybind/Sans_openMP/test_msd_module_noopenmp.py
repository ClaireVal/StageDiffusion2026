import numpy as np
import msd_module_noopenmp  # Importe le module compilé

# --- Exemple de données ---
# Positions aléatoires en 2D (1000 points, 2 dimensions)
N = 10000
dim = 2
x = np.array([(i,0) for i in range(N)])  # Tableau de forme (N, dim)

# Temps linéaire de 0 à 10
t = np.linspace(0, 10, N)   # Tableau de forme (N,)

# --- Appel de la fonction ---
dt_out, msd, dmsd = msd_module_noopenmp.MSDLongueVar_numba_opt(x, t)

# --- Affichage des résultats ---
print("Centres des bins (dt_out):")
print(dt_out)
print("\nValeurs du MSD:")
print(msd)
print("\nIncertitudes (dmsd):")
print(dmsd)

# --- Visualisation (optionnel) ---
import matplotlib.pyplot as plt

plt.errorbar(dt_out, msd, yerr=dmsd, fmt='o-', capsize=5, label="MSD")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Temps (dt)")
plt.ylabel("MSD")
plt.title("Mean Squared Displacement (MSD)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
