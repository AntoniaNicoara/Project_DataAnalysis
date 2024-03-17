import matplotlib.pyplot as plt
import numpy as np
from seaborn import heatmap

def show():
    plt.show()

#plot varianta
def plot_varianta(alpha, criterii, procent_minimal=80):
    fig = plt.figure("Plot varianta", figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title("Plot varianta", fontdict={"fontsize": 16, "color": "b"})
    ax.set_xlabel("Componenta")
    ax.set_ylabel("Varianta")
    x = np.arange(1, len(alpha) + 1)
    ax.set_xticks(x)
    ax.plot(x, alpha)
    ax.scatter(x, alpha, c="r", alpha=0.5)
    ax.axhline(alpha[criterii[0] - 1], c="m", label="Varianta minimala:" + str(procent_minimal) + "%")
    if not np.isnan(criterii[1]):
        ax.axhline(1, c="c", label="Kaiser")
    if not np.isnan(criterii[2]):
        ax.axhline(alpha[criterii[2] - 1], c="g", label="Cattell")
    ax.legend()
    plt.savefig("Plot_varianta")

#plot scoruri
def scatter(t,var1="C1",var2="C2",titlu="Plot scoruri",corelatii=False):
    fig = plt.figure(titlu+":"+var1+":"+var2, figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(titlu, fontdict={"fontsize": 16, "color": "b"})
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.axvline(0)
    ax.axhline(0)
    if corelatii:
        theta = np.arange(0,np.pi*2,0.01)
        ax.plot(np.cos(theta),np.sin(theta),c="m")
        ax.plot(0.65*np.cos(theta), 0.65*np.sin(theta), c="g")
    ax.scatter(t[var1],t[var2],c="b",alpha=0.5)
    for i in range(len(t)):
        ax.text(t[var1].iloc[i],t[var2].iloc[i],t.index[i])
    plt.savefig("Plot_scoruri")

def corelograma(t, vmin=-1, cmap="RdYlBu", annot=True, titlu="Corelatii factoriale"):
    fig = plt.figure(titlu, figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(titlu, fontdict={"fontsize": 16, "color": "b"})
    heatmap(t, vmin=vmin, vmax=1, cmap=cmap, annot=annot, ax=ax)
    plt.savefig("Corelograma_comunalitati")