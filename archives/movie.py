import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
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

XT = torch.Tensor(X).reshape(X.shape)
y = torch.Tensor(y).reshape(y.shape)
y_true = torch.Tensor(y_true).reshape(y_true.shape)

# Hypperparamètres

N = 500
l = 100
P0 = 0.1
epsilon = 0.1
sigma_0 = 0.5
fact = 0.1
ns = [1, 2, 1]
# tol = 1e-3

# Temperature initiale

T = 30

# Ici on commence l'algorithm BNN-ABC-SS

NP0 = int(N*P0)
invP0 = int(1/P0)
lll = BNN.modelSize(ns)

# Distance qui va être utiliser pour évaluer
# la dissimilarité entre les prédictions y_hat et la réponse y
pdist = 2

# L'a priopri gaussien N(0, I) pour les poids
thetas = torch.randn(lll, N)

# On evalue tout ça 
y_hats = torch.concat(
    tuple([BNN.FNN(ns, thetas[:, i]).forward(XT) for i in range(0, N)]), 1)

# On calcul la dissimilarité
rho_n = torch.cdist(y_hats.t(), y.t(), p=pdist)

# On fixe un petit truc qui va être utile après
sigma_j = sigma_0

# # Pour du débugage on ne fait pas attention
# # Relative learning rate
# lr = []
# stop = []


# Iteration
l_eps = []
for j in range(0, l):

    # On trie les erreurs et on mets les poids dans 
    # l'ordre croissant des érreurs qu'ils produisent
    rho_n, indices = torch.sort(rho_n, 0)
    thetas = thetas[:, indices.t()[0]]

    epsilon_j = rho_n[int(N*P0)]

    # # Lr
    # rho_nOld = rho_n

    # for debugging purposes
    l_eps.append(epsilon_j)

    thetasSeeds = (thetas[:, :NP0].t()).repeat(invP0, 1).t()
    rho_nSeeds = (rho_n[:NP0, 0].reshape(NP0, 1)).repeat(invP0, 1)

    # Resampling using seeds
    thetasResamples = torch.normal(thetasSeeds, sigma_j)

    # Evaluating performaces
    y_hatsResamples = torch.concat(
        tuple([BNN.FNN(ns, thetasResamples[:, i]).forward(XT) for i in range(0, N)]), 1)
    rho_nResamples = torch.cdist(y_hatsResamples.t(), y.t(), p=pdist)

    # Mask creation
    mask = torch.diag(
        (torch.bernoulli(1 / (1 + torch.exp(- (1/T) * (rho_nResamples - rho_n)))))[:, 0].float())

    thetas = torch.matmul(thetasResamples, mask) + \
        torch.matmul(thetasSeeds, torch.eye(N) - mask)
    rho_n = torch.matmul(mask, rho_nResamples) + \
        torch.matmul(torch.eye(N) - mask, rho_nSeeds)
    
    # sigma_j = (l - j + 1)*fact/l

    # Updating relative rate
    # lr.append(torch.cdist(rho_n.t(), rho_nOld.t()) / torch.norm(rho_n))

    # if(l > 2) :
    #     valStop = (lr[j] < tol) and (lr[j-1] < tol) and (lr[j-2] < tol)
    #     stop.append(valStop)
    #     if valStop :
    #         break
    plt.clf()
    BNN.plotTubeMedian(XT, y, thetas, ns)
    plt.show()
    plt.pause(0.1)


plt.clf()
BNN.plotTubeMedian(XT, y, thetas, ns)

# save the figure
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.pause(0.1)
