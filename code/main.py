import numpy as np
import pandas as pd

from functii import *
from grafice import *

np.set_printoptions(5, 10000, suppress=True)

set_date = pd.read_csv("input/tempo.csv", index_col=1)
variabile_observate = list(set_date)[1:]

nan_replace(set_date)

model_acp = acp(set_date, variabile_observate)
model_acp.fit()
# print("Varinta componentelor principale:")
# print(model_acp.alpha)

# Analiza variantei + plot varianta
varianta = model_acp.tabelare_varianta()
varianta.to_csv("Distributia_varianta.csv")
print(varianta)
print("Criterii de selectie:")
print(model_acp.criterii)
alpha = model_acp.alpha
plot_varianta(alpha, model_acp.criterii)

# Analiza corelatiilor factoriale
r = model_acp.r
etichete_componente = varianta.index
t_r = pd.DataFrame(r, variabile_observate, etichete_componente)
t_r.to_csv("Corelatii_var_obs_compPrin.csv")
corelograma(t_r)
show()
scatter(t_r,titlu="Plot Corelatii_var_obs_compPrin",corelatii=True)
show()

# Analiza scorurilor + plot scoruri
etichete_componente = varianta.index
c = model_acp.c
s = c / np.sqrt(np.maximum(alpha, 1e-10))
t_s = pd.DataFrame(s,set_date.index,etichete_componente)
t_s.to_csv("Scoruri.csv")
scatter(t_s)
show()

# Calcul cosinusuri
c2 = c * c
cosin = (c2.T / np.sum(c2, axis=1)).T
t_cosin = pd.DataFrame(cosin, set_date.index, etichete_componente)
t_cosin.to_csv("cosinusuri.csv")
cosin_max = t_cosin.apply(func=lambda x: x.index[x.argmax()], axis=1)
cosin_max.name = "Componenta dominanta"
cosin_max.to_csv("cosinusuri_max.csv")

# Contributiile
contrib = c2 * 100 / np.sum(c2, axis=0)
t_contrib = pd.DataFrame(contrib, set_date.index, etichete_componente)
t_contrib.to_csv("contributii.csv")
contrib_max = t_contrib.apply(func=lambda x: x.index[x.argmax()], axis=0)
contrib_max.name = "Instanta dominanta"
contrib_max.to_csv("contributii_max.csv")

# Comunalitati + corelograma
r = model_acp.r
r2 = r * r
comm = np.cumsum(r2, axis=1)
t_comm = pd.DataFrame(comm, variabile_observate, etichete_componente)
t_comm.to_csv("comunalitati.csv")
corelograma(t_comm,0,"Reds",titlu="Comunalitati")
show()

