import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier
df = pd.read_csv("D_traj_longue.csv")

plt.figure(figsize=(12, 8))  # Taille de la figure

# Grouper par alpha et tracer les données brutes
for alpha, data in df.groupby("Cdef"):
    plt.scatter(
        (alpha*np.ones(np.shape(data["D"]))),
        data["Alpha"],        
        c=((data["D"])),
        linewidth=3,  # Épaisseur des lignes
        s=80
    )

# plt.legend(prop={'size': 16}, frameon=True, fancybox=True, shadow=True)  # Légende
plt.ylabel("alpha", fontsize=18, labelpad=10)  # Label x
plt.xlabel("C_def", fontsize=18, labelpad=10)  # Label y
plt.grid(alpha=0.3)  # Grille discrète
plt.xticks(fontsize=18)  # Taille des ticks x
plt.yticks(fontsize=18)  # Taille des ticks y
plt.xscale("log")
plt.yscale("log")
plt.colorbar(label="D")
plt.tight_layout()  # Ajustement automatique
plt.savefig("etude_param.svg")
plt.show()
# plt.savefig("D_en_fonction_de_cDef_donnees_brutes.png", dpi=300, bbox_inches='tight')



plt.figure(figsize=(12, 8))  # Taille de la figure

# # Grouper par alpha et tracer les données brutes
# for alpha, data in df.groupby("Alpha"):
#     plt.plot(
#         data["Cdef"],
#         data["D"],
#         label=f"Alpha={alpha}",
#         linewidth=3,  # Épaisseur des lignes
#     )

# plt.legend(prop={'size': 16}, frameon=True, fancybox=True, shadow=True)  # Légende
# plt.xlabel("Concentration en défauts", fontsize=18, labelpad=10)  # Label x
# plt.ylabel("Coefficient de diffusion", fontsize=18, labelpad=10)  # Label y
# plt.grid(alpha=0.3)  # Grille discrète
# plt.xticks(fontsize=18)  # Taille des ticks x
# plt.yticks(fontsize=18)  # Taille des ticks y
# plt.xscale("log")
# plt.tight_layout()  # Ajustement automatique
# plt.show()
# # plt.savefig("D_en_fonction_de_cDef_donnees_brutes.png", dpi=300, bbox_inches='tight')
# # Grouper par Cdef
# fig, ax = plt.subplots(figsize=(12, 8))  # Taille de la figure

# for cD, data in df.groupby("Cdef"):
#     ax.plot(
#         data["Alpha"],
#         data["D"],
#         label=f"cDef={cD}",
#         linewidth=3,  # Épaisseur des lignes
#     )

# ax.set_xlabel("Alpha", fontsize=18, labelpad=10)  # Label x
# ax.set_ylabel("Coefficient de diffusion", fontsize=16, labelpad=10)  # Label y
# ax.set_xscale("log")  # Échelle logarithmique pour x
# # ax.set_yscale("log")  # Échelle logarithmique pour y
# ax.legend(prop={'size': 18}, frameon=True, fancybox=True, shadow=True)  # Légende
# ax.grid(alpha=0.3)  # Grille discrète
# ax.tick_params(axis='both', which='major', labelsize=14)  # Taille des ticks

# plt.tight_layout()  # Ajustement automatique
# plt.show()
# # plt.savefig("D_en_fonction_de_alpha_donnees_brutes.png", dpi=300, bbox_inches='tight')
