import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datetime import datetime
from utils import BNN

plt.ion()

# Un jour on saura coder sur GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Definition de la fonction de test
def f(x, sigma):
    epsilon = np.random.randn(*x.shape) * sigma
    return 10*np.sin(2*np.pi*x) + epsilon

# def f(x, sigma):
#     epsilon = np.random.randn(*x.shape) * sigma
#     return np.cos(x) + epsilon


# Paramétrage des données d'entrainements
train_size = 100
noise = 0.1
# Pour sin
xmin = -0.5
xmax = 0.5

# # Pour cos
# xmin = -3.5
# xmax = 3.5

X = np.linspace(xmin, xmax, train_size).reshape(-1, 1)
# X = np.linspace(-np.pi, np.pi, train_size).reshape(-1, 1)

y = f(X, sigma=noise)
y_true = f(X, sigma=0.0)


# On affiche tout ça
plt.scatter(X, y, marker='+', label='Training data')
plt.plot(X, y_true, label='Truth')
plt.title('Noisy training data and ground truth')
plt.legend()
plt.show()

XT = torch.Tensor(X).reshape(X.shape)
y = torch.Tensor(y).reshape(y.shape)
y_true = torch.Tensor(y_true).reshape(y_true.shape)

# Hypperparamètres
N = 1200
lmax = 200
P0 = 0.25
epsilon = 0.1
sigma_0 = 0.5
ns = [1, 3, 3, 1]

# Temperature initiale
# T = np.sum(1 / np.log(2 + np.arange(lmax))) + 15
T = 40
Tmin = 15

# Distance qui va être utiliser pour évaluer
# la dissimilarité entre les prédictions y_hat et la réponse y
pdist = 2

# Ici on commence l'algorithm BNN-ABC-SS
NP0 = int(N*P0)
invP0 = int(1/P0)
lll = BNN.modelSize(ns)


# L'a priopri gaussien N(0, I) pour les poids
thetas = torch.randn(lll, N)

# On evalue tout ça
y_hats = torch.concat(
    tuple([BNN.FNN(ns, thetas[:, i]).forward(XT) for i in range(0, N)]), 1)

# On calcul la dissimilarité
rho_n = torch.cdist(y_hats.t(), y.t(), p=pdist)

# On fixe un petit truc qui va être utile après
# c'est en fait le pas
sigma_j = sigma_0

# # Pour du débugage on ne fait pas attention
# # Relative learning rate
# lr = []
# stop = []


# Iteration
l_eps = []
Tcur = T
# for j in range(0, l):
j = 0
while (rho_n[0, 0] > epsilon):

    # On trie les erreurs et on mets les poids dans
    # l'ordre croissant des érreurs qu'ils produisent
    rho_n, indices = torch.sort(rho_n, 0)
    thetas = thetas[:, indices.t()[0]]

    epsilon_j = rho_n[NP0]

    # # Lr
    # rho_nOld = rho_n

    # Ici on a un échantillion de taille NP0 et on veut
    # en créer N à partir de cette échantillion en fesant
    # (invPO - 1) pas
    thetasSeeds = thetas[:, :NP0]
    rhoSeeds = rho_n[:NP0]

    # Réglage de sigma_j
    sigma_j = sigma_0

    #
    thetas = thetasSeeds
    rho_n = rho_n[:NP0]

    for g in range(invP0 - 1):
        # for debugging purposes
        l_eps.append(epsilon_j)
        thetasResamples = torch.normal(thetasSeeds, sigma_j)

        # On evalue les erreurs
        y_hatsResamples = torch.concat(
            tuple([BNN.FNN(ns, thetasResamples[:, i]).forward(XT) for i in range(0, NP0)]), 1)
        rhoResamples = torch.cdist(y_hatsResamples.t(), y.t(), p=pdist)

        # On évalue l'amélioration
        deltaRho = rhoResamples - rhoSeeds
        deltaTheta = thetasResamples - thetasSeeds

        # On voit si on avance dans le sens du gradient ou pas
        # Si il y a une amélioration on avance
        # Sinon, on remonte avec une proba de 1/ 1 + exp( - (1/ Tcur) * Grad)
        avanceOuNon = torch.bernoulli(
            1 / (1 + torch.exp(- (1/Tcur) * (deltaRho))))
        avanceOuNon[deltaRho < 0] = 0

        # ThetaFinal = ThetaResample - mask * (ThetaResample - ThetaSeed)
        mask = torch.diag(avanceOuNon[:, 0].float())

        # Mise à jour
        thetasSeeds = thetasResamples - torch.matmul(deltaTheta, mask)
        thetas = torch.concatenate((thetas, thetasSeeds), 1)
        rhoSeeds = rhoResamples - torch.matmul(mask, deltaRho)
        rho_n = torch.concatenate((rho_n, rhoSeeds))

    # Réglage de la température
    Tcur = max(T / np.log(2 + j), Tmin)
    j += 1
    if (j >= lmax):
        break

    plt.clf()
    BNN.plotTubeMedian(XT, y, thetas, ns)
    plt.title("Epoch {}| T {}".format(j, Tcur))
    plt.show()
    plt.pause(0.1)

plt.clf()
BNN.plotTubeMedian(XT, y, thetas, ns)
plt.title("Epoch {}".format(j))

# save the figure

# datetime object containing current date and time
now = datetime.now()
# dd-mm-YY_H:M:S
dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")

plt.savefig('plots/plot{}.png'.format(dt_string), dpi=300, bbox_inches='tight')
plt.show()
plt.pause(0.1)
