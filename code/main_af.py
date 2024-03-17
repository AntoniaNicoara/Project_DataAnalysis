import numpy as np

from functii import *
from grafice import *
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo

np.set_printoptions(5, 10000, suppress=True)

set_date = pd.read_csv("input/tempo.csv", index_col=1)
variabile_observate = list(set_date)[1:]

nan_replace(set_date)

x = set_date[variabile_observate].values

# Factorabilitatea datelor
test_bartlett = calculate_bartlett_sphericity(x)
print(test_bartlett)
if test_bartlett[1] > 0.001:
    print("Nu exista factori comuni!")
    exit(0)
kmo = calculate_kmo(x)
print(kmo)
t_kmo = pd.DataFrame(
    {
        "KMO": np.append(kmo[0], kmo[1])
    }, index=variabile_observate + ["KMO Total"]
)
t_kmo.to_csv("kmo.csv")
corelograma(t_kmo, 0, "Blues", titlu="Index KMO")
show()

# Construire model
n, m = x.shape

#CU ROTATIE
model_af = FactorAnalyzer(m, rotation="varimax")
model_af.fit(x)

# Analiza variantei + plot varianta
varianta = model_af.get_factor_variance()
# print(varianta)
etichete_factori = ["F" + str(i) for i in range(1, m + 1)]
tabel_varianta = pd.DataFrame(
    {
        "Varianta": varianta[0],
        "Procent varianta": varianta[1] * 100,
        "Procent cumulat": varianta[2] * 100
    }, etichete_factori
)
tabel_varianta.to_csv("varianta_factori.csv")
alpha = varianta[0]
criterii = calcul_criterii(alpha)
#plot_varianta(alpha, criterii, procent_minimal=70, eticheta_x="Factor")
show()

# Numar factori semnificativi
# k = np.nanmin(criterii)
k = int(np.nanmean(criterii))

# Analiza corelatiilor variabile-factori
l = model_af.loadings_
t_l = pd.DataFrame(l, variabile_observate, etichete_factori)
t_l.to_csv("corelatii_factoriale_rotatie.csv")
corelograma(t_l)
for j in range(2, k + 1):
    scatter(t_l, "F1", "F" + str(j), "Plot corelatii factoriale cu rotatie", corelatii=True)

# Calculul scorurilor
f = model_af.transform(x)
t_f = pd.DataFrame(f, set_date.index, etichete_factori)
t_f.to_csv("scoruri_factoriale_rotatie.csv")
for j in range(2, k + 1):
    scatter(t_f, "F1", "F" + str(j))

# Analiza comunalitatii si a variantei specifice
h = model_af.get_communalities()
psi = model_af.get_uniquenesses()
t_varianta_extrasa = pd.DataFrame(
    {
        "Comunalitate":h,
        "Varianta specifica":psi
    }, variabile_observate
)
t_varianta_extrasa.to_csv("comunalitati_factoriale.csv")
corelograma(t_varianta_extrasa,0,"Reds",titlu="Varianta extrasa")
show()

#FARA ROTATIE
model_af_none = FactorAnalyzer(m, rotation=None)
model_af_none.fit(x)

# Analiza variantei + plot varianta
varianta = model_af_none.get_factor_variance()
print(varianta)
etichete_factori = ["F" + str(i) for i in range(1, m + 1)]
tabel_varianta = pd.DataFrame(
    {
        "Varianta": varianta[0],
        "Procent varianta": varianta[1] * 100,
        "Procent cumulat": varianta[2] * 100
    }, etichete_factori
)
tabel_varianta.to_csv("varianta_af_fara_rotatie.csv")
alpha = varianta[0]
criterii = calcul_criterii(alpha)
#plot_varianta(alpha, criterii, procent_minimal=70, eticheta_x="Factor")
show()

# Numar factori semnificativi
# k = np.nanmin(criterii)
k = int(np.nanmean(criterii))

# Analiza corelatiilor variabile-factori
l = model_af_none.loadings_
t_l = pd.DataFrame(l, variabile_observate, etichete_factori)
t_l.to_csv("corelatii_factoriale_fara_rotatie.csv")
corelograma(t_l)
for j in range(2, k + 1):
    scatter(t_l, "F1", "F" + str(j), "Plot corelatii factoriale", corelatii=True)
show()

# Calculul scorurilor
f = model_af_none.transform(x)
t_f = pd.DataFrame(f, set_date.index, etichete_factori)
t_f.to_csv("scoruri_fara_rotatie.csv")
for j in range(2, k + 1):
    scatter(t_f, "F1", "F" + str(j))
show()




